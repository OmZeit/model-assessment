# tokenizer.py

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple
import gzip
import os

import torch
from tokenizers import Tokenizer

from .config import KMER_PAD_ID


BASE_TOKEN_IDS = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
}
BASE_ID_TO_TOKEN = {idx: base for base, idx in BASE_TOKEN_IDS.items()}

_RC = str.maketrans(
    {
        "A": "T",
        "C": "G",
        "G": "C",
        "T": "A",
        "a": "t",
        "c": "g",
        "g": "c",
        "t": "a",
        "N": "N",
        "n": "n",
    }
)

_KMER_BASE_TO_ID = {"A": 0, "C": 1, "G": 2, "T": 3}
_VALID_DNA = set("ACGTN")


def reverse_complement(seq: str) -> str:
    """Return reverse complement of a DNA sequence."""
    return seq.translate(_RC)[::-1]


def normalize_dna_sequence(seq: str) -> str:
    """Uppercase a DNA string and map non-ACGTN symbols to N without changing length."""
    return "".join(ch if ch in _VALID_DNA else "N" for ch in seq.upper())


def _open_text(path: str):
    """Open text file, handling gzip compression automatically."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "rt", encoding="utf-8", errors="ignore")


def read_fasta_records(path: str, chunk_size: Optional[int] = None) -> Iterable[Tuple[str, str, bool]]:
    """
    Minimal FASTA reader.

    Yields ``(header, sequence_or_chunk, is_last)``.  When ``chunk_size`` is
    provided, large records may be yielded in multiple chunks with ``is_last``
    marking the end of the original FASTA record.
    """
    hdr = None
    seq_parts: List[str] = []
    current_len = 0
    pending_chunk: Optional[str] = None

    def flush_record():
        nonlocal pending_chunk, seq_parts, current_len
        if hdr is None:
            return
        if chunk_size:
            if pending_chunk is not None:
                if seq_parts:
                    yield hdr, pending_chunk, False
                    yield hdr, "".join(seq_parts), True
                else:
                    yield hdr, pending_chunk, True
            elif seq_parts:
                yield hdr, "".join(seq_parts), True
        elif seq_parts:
            yield hdr, "".join(seq_parts), True
        pending_chunk = None
        seq_parts = []
        current_len = 0

    with _open_text(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                yield from flush_record()
                hdr = line[1:].strip()
            else:
                seq_parts.append(line)
                current_len += len(line)

                if chunk_size and current_len >= chunk_size:
                    chunk = "".join(seq_parts)
                    if pending_chunk is not None:
                        yield hdr or "", pending_chunk, False
                    pending_chunk = chunk
                    seq_parts = []
                    current_len = 0

        yield from flush_record()


def kmer_to_id(kmer: str) -> int:
    """Map an A/C/G/T k-mer to 1..4**k. 0 is reserved for pad/unknown."""
    value = 0
    for ch in kmer.upper():
        base = _KMER_BASE_TO_ID.get(ch)
        if base is None:
            return KMER_PAD_ID
        value = value * 4 + base
    return value + 1


class DnaTokenizer:
    def __init__(
        self,
        vocab_path: str = "dna_model/bpe_vocab/tokenizer.json",
        max_length: int = 8192,
        k_mer_sizes: Optional[Sequence[int]] = None,
        kmer_size: int = 3,
        return_kmer_ids: bool = True,
        tokenizer_mode: str = "bpe",
        **_: object,
    ):
        """
        Lightweight wrapper around the DNA token stream.

        ``tokenizer_mode="bpe"`` preserves the existing Hugging Face BPE stream
        and optional aligned 3-mer side channel.  ``tokenizer_mode="base"`` uses
        a strict four-token A/C/G/T stream with no special tokens in the model
        vocabulary.
        """
        self.max_length = int(max_length)
        mode = str(tokenizer_mode).lower()
        if mode not in {"bpe", "base"}:
            raise ValueError(f"tokenizer_mode must be 'bpe' or 'base', got {tokenizer_mode!r}")
        self.tokenizer_mode = mode
        self.vocab_path = vocab_path
        if k_mer_sizes:
            self.k_mer_sizes = [int(k) for k in k_mer_sizes]
            self.kmer_size = self.k_mer_sizes[0]
        else:
            self.k_mer_sizes = [int(kmer_size)]
            self.kmer_size = int(kmer_size)
        self.k_mer_size = self.kmer_size  # legacy attribute name
        self.return_kmer_ids = bool(return_kmer_ids) and self.tokenizer_mode == "bpe"
        self.kmer_pad_id = KMER_PAD_ID
        self.kmer_vocab_size = (4 ** self.kmer_size) + 1
        self.vocab_sizes = {int(k): (4 ** int(k)) + 1 for k in self.k_mer_sizes}

        if self.tokenizer_mode == "base":
            self.special_ids = {}
            self.pad_token_id = None
            self.unk_token_id = None
            self.cls_token_id = None
            self.sep_token_id = None
            self.mask_token_id = None
            self.special_tokens_max_id = -1
            self.tokenizer = None
            self.vocab_size = len(BASE_TOKEN_IDS)
            return

        self.special_ids = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[CLS]": 2,
            "[SEP]": 3,
            "[MASK]": 4,
        }
        self.pad_token_id = self.special_ids["[PAD]"]
        self.unk_token_id = self.special_ids["[UNK]"]
        self.cls_token_id = self.special_ids["[CLS]"]
        self.sep_token_id = self.special_ids["[SEP]"]
        self.mask_token_id = self.special_ids["[MASK]"]
        self.special_tokens_max_id = max(self.special_ids.values())

        if not os.path.exists(vocab_path) and not os.path.isabs(vocab_path):
            module_dir = os.path.dirname(os.path.abspath(__file__))
            rel = vocab_path.replace("\\", "/")
            if rel.startswith("dna_model/"):
                rel = rel[len("dna_model/") :]
            alt_path = os.path.join(module_dir, rel)
            if os.path.exists(alt_path):
                vocab_path = alt_path

        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"BPE tokenizer not found at {vocab_path}. Run scripts/train_bpe.py first.")

        self.tokenizer = Tokenizer.from_file(vocab_path)

        required_specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        for token in required_specials:
            tid = self.tokenizer.token_to_id(token)
            if tid is None:
                raise ValueError(f"Special token '{token}' missing from BPE vocab at {vocab_path}")
            self.special_ids[token] = tid

        self.pad_token_id = self.special_ids["[PAD]"]
        self.unk_token_id = self.special_ids["[UNK]"]
        self.cls_token_id = self.special_ids["[CLS]"]
        self.sep_token_id = self.special_ids["[SEP]"]
        self.mask_token_id = self.special_ids["[MASK]"]
        self.special_tokens_max_id = max(self.special_ids.values())
        self.vocab_size = self.tokenizer.get_vocab_size()

    def _encode_base(
        self,
        sequence: str,
        *,
        max_length: Optional[int],
        return_tensors: Optional[str],
        padding: bool,
        truncation: bool,
        return_offsets: bool,
    ):
        effective_max = int(max_length or self.max_length)
        if effective_max <= 0:
            raise ValueError(f"max_length must be positive, got {effective_max}")

        input_ids = []
        offsets = []
        attention_mask = []
        token_budget = effective_max if truncation else len(sequence)
        for idx, base in enumerate(sequence[:token_budget]):
            if base in BASE_TOKEN_IDS:
                input_ids.append(BASE_TOKEN_IDS[base])
                attention_mask.append(1)
            else:
                # Handle 'N' or any non-ACGT character by mapping to 'A' (0) 
                # and explicitly setting attention_mask to 0 so it's ignored by the model.
                input_ids.append(BASE_TOKEN_IDS["A"])
                attention_mask.append(0)
            offsets.append((idx, idx + 1))

        if padding and len(input_ids) < effective_max:
            pad_len = effective_max - len(input_ids)
            input_ids.extend([BASE_TOKEN_IDS["A"]] * pad_len)
            attention_mask.extend([0] * pad_len)
            offsets.extend([(0, 0)] * pad_len)

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if return_offsets:
            out["offsets"] = offsets
        if return_tensors == "pt":
            return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}
        return out

    def kmer_ids_from_offsets(
        self,
        sequence: str,
        offsets: Sequence[Tuple[int, int]],
        *,
        target_length: Optional[int] = None,
    ) -> List[int]:
        ids: List[int] = []
        seq_upper = sequence.upper()
        k = self.kmer_size
        for start, end in offsets:
            if end <= start:
                ids.append(self.kmer_pad_id)
                continue
            token_seq = seq_upper[start:end]
            if len(token_seq) < k:
                ids.append(self.kmer_pad_id)
                continue
            ids.append(kmer_to_id(token_seq[:k]))

        if target_length is not None:
            if len(ids) > target_length:
                ids = ids[:target_length]
            elif len(ids) < target_length:
                ids.extend([self.kmer_pad_id] * (target_length - len(ids)))
        return ids

    def encode(
        self,
        sequence: str,
        *,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = "pt",
        padding: bool = True,
        truncation: bool = True,
        return_kmer_ids: Optional[bool] = None,
        return_offsets: bool = False,
    ):
        """Encode a raw DNA sequence into token IDs and optional metadata."""
        sequence = normalize_dna_sequence(sequence)
        effective_max = int(max_length or self.max_length)
        if self.tokenizer_mode == "base":
            return self._encode_base(
                sequence,
                max_length=effective_max,
                return_tensors=return_tensors,
                padding=padding,
                truncation=truncation,
                return_offsets=return_offsets,
            )

        encoded = self.tokenizer.encode(sequence)

        input_ids = list(encoded.ids)
        attention_mask = list(encoded.attention_mask)
        offsets = list(encoded.offsets)
        include_kmers = self.return_kmer_ids if return_kmer_ids is None else bool(return_kmer_ids)

        if truncation and len(input_ids) > effective_max:
            input_ids = input_ids[:effective_max]
            attention_mask = attention_mask[:effective_max]
            offsets = offsets[:effective_max]
            if input_ids and input_ids[-1] != self.sep_token_id:
                input_ids[-1] = self.sep_token_id
                attention_mask[-1] = 1
                offsets[-1] = (0, 0)

        kmer_ids = None
        if include_kmers:
            kmer_ids = self.kmer_ids_from_offsets(sequence, offsets)

        if padding and len(input_ids) < effective_max:
            pad_len = effective_max - len(input_ids)
            input_ids.extend([self.pad_token_id] * pad_len)
            attention_mask.extend([0] * pad_len)
            offsets.extend([(0, 0)] * pad_len)
            if kmer_ids is not None:
                kmer_ids.extend([self.kmer_pad_id] * pad_len)

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if kmer_ids is not None:
            out["kmer_ids"] = kmer_ids
        if return_offsets:
            out["offsets"] = offsets

        if return_tensors == "pt":
            return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}
        return out

    def encode_with_kmers(self, sequence: str, max_length: Optional[int] = None, **kwargs):
        """Backward-compatible alias used by older scripts and utilities."""
        kwargs.setdefault("return_tensors", None)
        kwargs.setdefault("return_kmer_ids", self.tokenizer_mode == "bpe")
        return self.encode(sequence, max_length=max_length, **kwargs)

    def decode(self, token_ids, skip_special_tokens: bool = True):
        """Decode token IDs back into a continuous DNA sequence."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if self.tokenizer_mode == "base":
            bases = []
            for token_id in token_ids:
                token_id = int(token_id)
                if token_id in BASE_ID_TO_TOKEN:
                    bases.append(BASE_ID_TO_TOKEN[token_id])
            return "".join(bases)
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
