# dataset.py

import logging
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from .config import (
    PAD_TOKEN_ID,
    KMER_PAD_ID,
)
from .tokenizer import DnaTokenizer, read_fasta_records

LOGGER = logging.getLogger(__name__)


def _normalize_fasta_offsets(
    offsets: Sequence[Sequence[int]],
) -> List[Tuple[int, int, int, int]]:
    """Return FASTA offsets as ``(sequence_index, start, end, length)``.

    Older callers supplied three-tuples while the lazy reader expected four
    values. Accept both representations at the boundary and keep one internal
    shape so lazy loading cannot fail during iteration.
    """
    normalized: List[Tuple[int, int, int, int]] = []
    for position, raw in enumerate(offsets):
        values = tuple(int(value) for value in raw)
        if len(values) == 4:
            sequence_index, start, end, length = values
        elif len(values) == 3:
            start, end, length = values
            sequence_index = position
        else:
            raise ValueError(
                "FASTA offsets must contain three or four integers; "
                f"entry {position} has {len(values)} values."
            )
        normalized.append((sequence_index, start, end, length))
    return normalized


# Place this at the top of dataset.py
rc_translation_table = str.maketrans('ACGTNacgtn', 'TGCANtgcan')

def fast_reverse_complement(seq_string: str) -> str:
    """Return the O(N) reverse complement of a raw DNA string.

    Args:
        seq_string: DNA sequence over A/C/G/T/N. Case is preserved for matching
            lowercase bases.

    Returns:
        Reverse-complemented DNA sequence with unchanged length.
    """
    return seq_string.translate(rc_translation_table)[::-1]


_VALID_BASES = "ACGTN"
_INSERTABLE_BASES = "ACGT"


def apply_frameshift_perturbation(seq: str, rng: random.Random) -> str:
    """
    Pre-BPE single-base indel perturbation.

    Randomly inserts or deletes a single base in the sequence with equal
    probability.  Operating on the raw nucleotide string *before* BPE
    tokenization introduces a biologically plausible local perturbation.
    Because BPE tokens are variable-length, downstream token-level effects
    are intentionally stochastic (merge boundaries may shift by a little or a lot).

    Args:
        seq: Raw DNA string (ACGTN alphabet).
        rng: A seeded :class:`random.Random` instance for reproducibility.

    Returns:
        The perturbed sequence string, or the original if it is too short.
    """
    if len(seq) < 2:
        return seq

    if rng.random() < 0.5:
        # Insertion: pick a random position and insert a random base
        pos = rng.randint(0, len(seq))
        base = rng.choice(_INSERTABLE_BASES)
        seq = seq[:pos] + base + seq[pos:]
    else:
        # Deletion: remove one random base
        pos = rng.randint(0, len(seq) - 1)
        seq = seq[:pos] + seq[pos + 1:]

    return seq

def collate_bpe(batch: List[Dict[str, torch.Tensor]], max_length: int = None) -> Dict[str, torch.Tensor]:
    """
    Collate function for BPE tokens, strictly ignoring legacy kmer_ids.
    """
    if not batch:
        return {}
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    
    # Pad input_ids and attention_mask
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=PAD_TOKEN_ID)
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    
    if max_length is not None:
        curr_len = input_ids_padded.shape[1]
        if curr_len < max_length:
             pad_amt = max_length - curr_len
             input_ids_padded = torch.nn.functional.pad(input_ids_padded, (0, pad_amt), value=PAD_TOKEN_ID)
             attention_mask_padded = torch.nn.functional.pad(attention_mask_padded, (0, pad_amt), value=0)

    out = {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded
    }

    if "kmer_ids" in batch[0]:
        kmer_ids = [item["kmer_ids"] for item in batch]
        out["kmer_ids"] = pad_sequence(kmer_ids, batch_first=True, padding_value=KMER_PAD_ID)
        if max_length is not None and out["kmer_ids"].shape[1] < max_length:
            pad_amt = max_length - out["kmer_ids"].shape[1]
            out["kmer_ids"] = torch.nn.functional.pad(out["kmer_ids"], (0, pad_amt), value=KMER_PAD_ID)
    
    # Handle RC-pair consistency tokens if provided
    if "input_ids_rc" in batch[0]:
        ids_rc = [item["input_ids_rc"] for item in batch]
        masks_rc = [item["attention_mask_rc"] for item in batch]
        out["input_ids_rc"] = pad_sequence(ids_rc, batch_first=True, padding_value=PAD_TOKEN_ID)
        out["attention_mask_rc"] = pad_sequence(masks_rc, batch_first=True, padding_value=0)
        if "kmer_ids_rc" in batch[0]:
            kmers_rc = [item["kmer_ids_rc"] for item in batch]
            out["kmer_ids_rc"] = pad_sequence(kmers_rc, batch_first=True, padding_value=KMER_PAD_ID)
        
    if "domain_name" in batch[0]:
        out["domain_name"] = batch[0]["domain_name"]
        
    return out


collate = collate_bpe


class DatasetImpl(Dataset):
    """
    Optimized dataset with minimum chunk length validation.
    """
    
    def __init__(
        self,
        tokenizer: DnaTokenizer,
        sequences_or_path,
        max_length,
        stride,
        use_reverse_prob=0.0,       # float prob in [0, 1]
        provide_rc=False,           # Whether to return (Original, RC) pairs for consistency
        min_chunk_length=None,
        indices: Optional[List[int]] = None,
        lazy_load: bool = False,
        precomputed_offsets: Optional[Sequence[Sequence[int]]] = None,
        frameshift_prob: float = 0.0,   # Pre-BPE frameshift perturbation probability (Step 3)
        seed: int = 0,
    ):
        self.tok = tokenizer
        self.max_length = int(max_length)
        self.stride = int(stride)
        self.use_reverse_prob = float(use_reverse_prob)
        self.provide_rc = provide_rc
        self.frameshift_prob = float(frameshift_prob)
        self.seed = int(seed)
        self.min_chunk_length = min_chunk_length or max(64, max_length // 4)

        # --- RC probability as a float (not boolean) ---
        self.use_reverse_prob = float(use_reverse_prob)
        self.use_reverse_prob = max(0.0, min(1.0, self.use_reverse_prob))  # clamp 
        self.frameshift_prob = max(0.0, min(1.0, self.frameshift_prob))  # clamp
        
        self.chunk_coords = []
        self.epoch = 0  # For deterministic augmentation
        self.file_path = None
        self.sequences = []
        self.indices = indices
        self.lazy_load = lazy_load
        self.seq_offsets = []  # List of (sequence_index, start_offset, end_offset, seq_len)
        self.worker_file_handle = None

        # Handle input: either a list of strings or a file path
        if isinstance(sequences_or_path, str) and os.path.exists(sequences_or_path):
            self.file_path = sequences_or_path
            if self.lazy_load:
                if precomputed_offsets is not None:
                    normalized_offsets = _normalize_fasta_offsets(precomputed_offsets)
                    if self.indices is not None:
                        self.seq_offsets = [normalized_offsets[i] for i in self.indices]
                    else:
                        self.seq_offsets = normalized_offsets
                    LOGGER.info("[DatasetImpl] Using precomputed offsets for %s sequences.", len(self.seq_offsets))
                else:
                    self._scan_file_offsets()
            else:
                # Load immediately for main process to build chunks
                self._load_sequences()
        else:
            self.sequences = list(sequences_or_path)
            if self.lazy_load:
                LOGGER.warning("[DatasetImpl] lazy_load=True but input is a list; ignoring lazy_load.")
                self.lazy_load = False

        # Build chunks (only happens once in main process)
        skipped_short = 0
        
        # Abstract over sequences vs offsets
        num_seqs = len(self.seq_offsets) if self.lazy_load else len(self.sequences)
        
        for i in range(num_seqs):
            if self.lazy_load:
                # We have length pre-calculated
                # FIX: Unpack 4 values (index, start, end, len)
                _, _, _, seq_len = self.seq_offsets[i]
            else:
                seq_len = len(self.sequences[i])
                
            if seq_len < self.min_chunk_length:
                skipped_short += 1
                continue
            if seq_len <= self.max_length:
                self.chunk_coords.append((i, 0, seq_len))

            else:
                # Generate chunks with stride, ensuring minimum length
                for s in range(0, seq_len - self.min_chunk_length + 1, max(1, self.stride)):
                    end = min(s + self.max_length, seq_len)
                    chunk_len = end - s
                    if chunk_len >= self.min_chunk_length:
                        self.chunk_coords.append((i, s, end))
                    if s + self.max_length >= seq_len:
                        break
        
        if skipped_short > 0:
            LOGGER.info("[DatasetImpl] Skipped %s sequences < %sbp", skipped_short, self.min_chunk_length)

    def _scan_file_offsets(self):
        """Scan FASTA file for sequence offsets and lengths."""
        LOGGER.info("[DatasetImpl] Scanning %s for offsets...", self.file_path)
        self.seq_offsets = []
        
        all_offsets = []
        
        with open(self.file_path, "rb") as f:
            offset = 0
            current_seq_start = -1
            current_seq_len = 0
            
            for line in f:
                line_len = len(line)
                if line.startswith(b">"):
                    if current_seq_start != -1:
                        all_offsets.append((len(all_offsets), current_seq_start, offset, current_seq_len))
                    
                    current_seq_start = offset + line_len
                    current_seq_len = 0
                else:
                    if current_seq_start != -1:
                        stripped = line.strip()
                        current_seq_len += len(stripped)
                
                offset += line_len
            
            if current_seq_start != -1:
                all_offsets.append((len(all_offsets), current_seq_start, offset, current_seq_len))
                
        if self.indices is not None:
            self.seq_offsets = [all_offsets[i] for i in self.indices]
        else:
            self.seq_offsets = all_offsets
            
        LOGGER.info("[DatasetImpl] Found %s sequences (lazy loaded).", len(self.seq_offsets))

    def _load_sequences(self):
        """Helper to load sequences from file."""
        if self.file_path:
            # We use the shared helper but discard headers
            all_seqs = [seq for _hdr, seq, _is_last in read_fasta_records(self.file_path)]
            if self.indices is not None:
                # Filter based on provided indices
                # We use a set for faster lookup if indices are not sorted, but list comprehension is fine
                # Assuming indices are valid for all_seqs
                self.sequences = [all_seqs[i] for i in self.indices]
            else:
                self.sequences = all_seqs

    def __getstate__(self):
        """Custom pickling: Don't pickle the huge sequences list if we have a file path."""
        state = self.__dict__.copy()
        if self.file_path:
            # Remove the heavy data, worker will reload it from file_path
            state['sequences'] = [] 
        # Don't pickle the file handle
        state['worker_file_handle'] = None
        return state

    def __setstate__(self, state):
        """Custom unpickling: Reload sequences from file if needed."""
        self.__dict__.update(state)
        if self.file_path and not self.sequences and not self.lazy_load:
            self._load_sequences()

    def set_epoch(self, epoch: int):
        """Set epoch for deterministic augmentation."""
        self.epoch = epoch



    def __len__(self):
        return len(self.chunk_coords)

    def get_lengths(self):
        """Return lengths of all chunks."""
        return [end - start for _, start, end in self.chunk_coords]
        
    def _ensure_handle(self):
        if self.worker_file_handle is None and self.file_path:
            self.worker_file_handle = open(self.file_path, "rb")

    def __getitem__(self, idx):
        seq_idx, start, end = self.chunk_coords[idx]
        
        # Retrieve sequence
        if self.lazy_load:
            self._ensure_handle()
            f = self.worker_file_handle
            
            # Get offset info
            # seq_offsets contains (index, start_offset, end_offset, seq_len)
            # FIX: Unpack 4 values
            _, seq_start_offset, seq_end_offset, _ = self.seq_offsets[seq_idx]
            f.seek(seq_start_offset)
            raw_bytes = f.read(seq_end_offset - seq_start_offset)
            # Remove newlines
            seq = raw_bytes.replace(b"\n", b"").replace(b"\r", b"").decode("utf-8")
        else:
            seq = self.sequences[seq_idx]
            
        # Slice
        subseq = seq[start:end]
        
        # Augmentation
        # Deterministic seed based on run seed + epoch + idx.
        rng = random.Random(self.seed + self.epoch * 1000000 + idx)
        
        if self.use_reverse_prob > 0 and rng.random() < self.use_reverse_prob:
            subseq = fast_reverse_complement(subseq)

        # Pre-BPE single-base indel perturbation
        # Applied BEFORE tokenisation so the BPE tokeniser sees the perturbed
        # nucleotide string with stochastic merge-boundary disruption.
        if self.frameshift_prob > 0.0 and rng.random() < self.frameshift_prob:
            subseq = apply_frameshift_perturbation(subseq, rng)
            
        # Tokenize (using BPE encode)
        enc = self.tok.encode(subseq, max_length=self.max_length)
        
        item = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }
        if "kmer_ids" in enc:
            item["kmer_ids"] = enc["kmer_ids"]

        # Handle RC Consistency (Option A: dual tokens)
        if self.provide_rc:
            # RC must be computed on raw nucleotide string BEFORE BPE
            subseq_rc = fast_reverse_complement(subseq)
            enc_rc = self.tok.encode(subseq_rc, max_length=self.max_length)
            item["input_ids_rc"] = enc_rc["input_ids"]
            item["attention_mask_rc"] = enc_rc["attention_mask"]
            if "kmer_ids" in enc_rc:
                item["kmer_ids_rc"] = enc_rc["kmer_ids"]
             
        # Add domain name if available (from file path)
        if self.file_path:
             item["domain_name"] = os.path.basename(self.file_path)
             
        return item


from .loaders import (
    NpyDataset,
    RoundRobinLoader,
    SmartBatchSampler,
    build_loaders_balanced,
    count_chunks_exact,
    scan_fasta_lengths,
)
