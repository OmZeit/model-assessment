from __future__ import annotations

from pathlib import Path

import pytest
import torch

from model_assessment.model_bundles.dnabert2_117m import runtime


class _FakeModel:
    def eval(self) -> None:
        pass

    def parameters(self) -> list[object]:
        return []

    def to(self, _device: torch.device) -> None:
        pass


def test_backbone_manifest_is_verified_before_remote_code_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    def verified(_path: Path) -> dict[str, str]:
        events.append("verified")
        return {"revision": runtime.PINNED_REVISION}

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            assert events == ["verified"]
            events.append("tokenizer")
            return object()

    class FakeConfig:
        attention_probs_dropout_prob = 0.0

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            assert events == ["verified", "tokenizer"]
            events.append("config")
            return cls()

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            assert events == ["verified", "tokenizer", "config"]
            events.append("model")
            return _FakeModel()

    monkeypatch.setattr(runtime, "verify_model_manifest", verified)
    monkeypatch.setattr(runtime, "AutoTokenizer", FakeTokenizer)
    monkeypatch.setattr(runtime, "BertConfig", FakeConfig)
    monkeypatch.setattr(runtime, "AutoModel", FakeAutoModel)

    runtime.load_frozen_backbone(tmp_path, torch.device("cpu"))

    assert events == ["verified", "tokenizer", "config", "model"]


def test_failed_manifest_verification_prevents_transformers_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runtime,
        "verify_model_manifest",
        lambda _path: (_ for _ in ()).throw(ValueError("tampered model")),
    )

    class ForbiddenLoader:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            raise AssertionError("Transformers loader ran before verification")

    monkeypatch.setattr(runtime, "AutoTokenizer", ForbiddenLoader)
    monkeypatch.setattr(runtime, "AutoModel", ForbiddenLoader)

    with pytest.raises(ValueError, match="tampered"):
        runtime.load_frozen_backbone(tmp_path, torch.device("cpu"))
