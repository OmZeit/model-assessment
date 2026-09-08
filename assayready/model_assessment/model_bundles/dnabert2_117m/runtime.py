from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from transformers.models.bert.configuration_bert import BertConfig


PINNED_REPOSITORY = "zhihan1996/DNABERT-2-117M"
PINNED_REVISION = "a3d38b3f41cec05e370a4d3eeb8664fcb4bce227"
PINNED_MODEL_FILES = {
    "LICENSE": (11_357, "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"),
    "README.md": (1_316, "48b18abd051eb4952e0c0a50a0740387a1cea145be06517b67ab73a29431a3bc"),
    "bert_layers.py": (40_690, "317ad7e9667980ac724c07f0174ba265c8446ca5230f6b473068ff1b911edcf9"),
    "bert_padding.py": (6_099, "44d1c68afb1f585fdc66c150d4c60f1ed44a89c006abc57d50531d71940d7421"),
    "config.json": (904, "ba9bdafaff0cc3e30556927474d4a179519a9864012bed2628e9f1bc23c84bfd"),
    "configuration_bert.py": (1_011, "95fc868641b87bbcd7a32d2cd7b9f4769c27592e129daf167d14b5b8c74ec4c5"),
    "flash_attn_triton.py": (42_737, "568d1ac3beca0b5e1df528a1f136aa19b6489a616fcf3784f33336a50bb1de81"),
    "generation_config.json": (90, "c993e393c12525ea015019130f83c90502efaa6de8d019e94856555614d9fae3"),
    "model.safetensors": (468_313_032, "bc91ac0d972a698b7ff12ea6815e966b1feff9d7d1bef3a10535f2f4332609ac"),
    "tokenizer.json": (167_908, "5d178e8ce2ba55df97fff197f4b30f40133b95d7096be398c2df6b526c5d8cd3"),
    "tokenizer_config.json": (158, "f9d18c81f4dd9dd7db02e9f27cc1203228147d890bfce9167c3af6465ff5b769"),
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_sequences(values: Iterable[object]) -> list[str]:
    sequences = []
    for index, value in enumerate(values):
        sequence = "".join(str(value or "").upper().split())
        invalid = sorted(set(sequence) - set("ACGTN"))
        if not sequence or invalid:
            raise ValueError(f"invalid DNA sequence at row {index}: invalid symbols={invalid}")
        sequences.append(sequence)
    return sequences


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def load_frozen_backbone(model_dir: str | Path, device: torch.device):
    model_root = Path(model_dir).resolve()
    # This must precede every Transformers call because trust_remote_code loads
    # and executes Python modules from the snapshot.
    verify_model_manifest(model_root)
    model_path = str(model_root)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True
    )
    config = BertConfig.from_pretrained(model_path, local_files_only=True)
    # DNABERT-2's optional 2023 Triton attention kernel predates Blackwell.
    # A nonzero configured value selects its standard PyTorch attention path;
    # eval mode still disables dropout, so embeddings remain deterministic.
    config.attention_probs_dropout_prob = 1e-8
    # The pinned remote module catches ImportError and selects its portable
    # attention path when Triton is absent. Temporarily mask Triton so merely
    # importing its old optional kernel cannot trigger JIT compiler setup.
    missing = object()
    previous_triton = sys.modules.get("triton", missing)
    sys.modules["triton"] = None
    try:
        model = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            config=config,
            use_safetensors=True,
        )
    finally:
        if previous_triton is missing:
            sys.modules.pop("triton", None)
        else:
            sys.modules["triton"] = previous_triton
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device)
    return tokenizer, model


def embed_sequences(
    sequences: list[str],
    *,
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    embeddings = []
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start : start + batch_size]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
            with torch.autocast(
                device_type="cuda", dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                hidden = model(**encoded)[0]
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            embeddings.append(pooled.float().cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def verify_model_manifest(model_dir: str | Path) -> dict:
    root = Path(model_dir)
    manifest_path = root / "assayready_model_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("model directory lacks assayready_model_manifest.json; use the pinned downloader.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("repository") != PINNED_REPOSITORY
        or manifest.get("revision") != PINNED_REVISION
    ):
        raise ValueError("model manifest is not the pinned AssayReady DNABERT-2 revision.")
    for name, (expected_size, expected_digest) in PINNED_MODEL_FILES.items():
        path = root / name
        if not path.is_file():
            raise ValueError(f"pinned DNABERT-2 snapshot is missing {name!r}.")
        if path.stat().st_size != expected_size:
            raise ValueError(f"pinned DNABERT-2 file {name!r} has the wrong size.")
        if sha256_file(path) != expected_digest:
            raise ValueError(f"pinned DNABERT-2 file {name!r} failed SHA-256 verification.")
    return manifest
