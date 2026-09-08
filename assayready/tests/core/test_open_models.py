import json
from pathlib import Path

import pytest

import model_assessment.open_models as open_models
from model_assessment.open_models import (
    DNABERT2_REVISION,
    DNABERT2_WEIGHTS_SHA256,
    download_dnabert2,
    dnabert2_lock,
    ensure_dnabert2_scaffold,
    scaffold_dnabert2_bundle,
    verify_manifest_file_records,
)


def _file_record(path: Path, *, name: str | None = None) -> dict[str, object]:
    return {
        "name": name or path.name,
        "size_bytes": path.stat().st_size,
        "sha256": open_models._sha256(path),
    }


def test_dnabert2_choice_is_immutable_safe_and_commercially_usable() -> None:
    lock = dnabert2_lock()
    assert len(DNABERT2_REVISION) == 40
    assert len(DNABERT2_WEIGHTS_SHA256) == 64
    assert lock["license"] == "Apache-2.0"
    assert lock["parameters"] == 117_000_000
    assert lock["weights_file"] == "model.safetensors"


def test_scaffold_creates_complete_offline_bundle_without_weights(tmp_path: Path) -> None:
    bundle = scaffold_dnabert2_bundle(tmp_path / "bundle")
    expected = {
        "Dockerfile", "README.md", "runtime.py", "train_head.py", "predict.py",
        "requirements.lock.txt", "controlled_execution.json", "model.lock.json",
        "weights", "task_head",
    }
    assert expected <= {item.name for item in bundle.iterdir()}
    lock = json.loads((bundle / "model.lock.json").read_text(encoding="utf-8"))
    assert lock["revision"] == DNABERT2_REVISION
    assert "latest" not in (bundle / "Dockerfile").read_text(encoding="utf-8").lower()
    assert "HF_HUB_OFFLINE=1" in (bundle / "Dockerfile").read_text(encoding="utf-8")
    train_head = (bundle / "train_head.py").read_text(encoding="utf-8")
    assert "--objective-direction" in train_head
    assert '"objective_direction": objective_direction' in train_head
    assert "objective_direction=args.objective_direction" in train_head
    runtime = (bundle / "runtime.py").read_text(encoding="utf-8")
    assert "PINNED_MODEL_FILES" in runtime
    assert '"configuration_bert.py"' in runtime
    assert not any((bundle / "weights").iterdir())


def test_scaffold_refuses_to_overwrite_user_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "user-file").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        scaffold_dnabert2_bundle(bundle)


def test_prepare_scaffold_is_rerunnable_and_restores_only_missing_files(tmp_path: Path) -> None:
    bundle = scaffold_dnabert2_bundle(tmp_path / "bundle")
    custom = bundle / "operator-notes.txt"
    custom.write_text("keep me", encoding="utf-8")
    missing = bundle / "predict.py"
    missing.unlink()

    assert ensure_dnabert2_scaffold(bundle) == bundle.resolve()
    assert missing.is_file()
    assert custom.read_text(encoding="utf-8") == "keep me"


def test_prepare_scaffold_refuses_unrecognized_nonempty_directory(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "user-model.bin").write_bytes(b"owned by the user")

    with pytest.raises(FileExistsError, match="unrecognized non-empty"):
        ensure_dnabert2_scaffold(bundle)


def test_manifest_file_records_verify_size_digest_and_nested_paths(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    nested = model_dir / "tokenizer" / "config.json"
    nested.parent.mkdir(parents=True)
    nested.write_text('{"test": true}', encoding="utf-8")
    manifest = {"files": [_file_record(nested, name="tokenizer/config.json")]}

    verified = verify_manifest_file_records(model_dir, manifest)

    assert verified["tokenizer/config.json"][0] == nested.resolve()
    nested.write_text('{"test": null}', encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_manifest_file_records(model_dir, manifest)


@pytest.mark.parametrize("unsafe_name", ["../outside.bin", "..\\outside.bin", "/outside.bin", "C:\\outside.bin"])
def test_manifest_file_records_reject_path_escape(tmp_path: Path, unsafe_name: str) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    manifest = {"files": [_file_record(outside, name=unsafe_name)]}

    with pytest.raises(ValueError, match="relative|unsafe path component"):
        verify_manifest_file_records(model_dir, manifest)


def test_cached_download_fast_path_verifies_every_manifest_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_dir = tmp_path / "weights"
    model_dir.mkdir()
    weights = model_dir / "model.safetensors"
    auxiliary = model_dir / "config.json"
    weights.write_bytes(b"test-weights")
    auxiliary.write_bytes(b"model-config")
    manifest = {
        "revision": DNABERT2_REVISION,
        "files": [_file_record(weights), _file_record(auxiliary)],
    }
    (model_dir / "assayready_model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(open_models, "DNABERT2_WEIGHTS_SIZE", weights.stat().st_size)
    monkeypatch.setattr(open_models, "DNABERT2_WEIGHTS_SHA256", open_models._sha256(weights))
    monkeypatch.setattr(
        open_models,
        "DNABERT2_PINNED_FILES",
        {
            "model.safetensors": (weights.stat().st_size, open_models._sha256(weights)),
            "config.json": (auxiliary.stat().st_size, open_models._sha256(auxiliary)),
        },
    )

    assert download_dnabert2(model_dir) == manifest

    auxiliary.write_bytes(b"tamper-confg")
    with pytest.raises(ValueError, match="existing model directory.*SHA-256 mismatch"):
        download_dnabert2(model_dir)


def test_pinned_snapshot_rejects_self_consistent_but_untrusted_remote_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "weights"
    model_dir.mkdir()
    remote_code = model_dir / "configuration_bert.py"
    remote_code.write_text("# expected\n", encoding="utf-8")
    expected = (remote_code.stat().st_size, open_models._sha256(remote_code))
    monkeypatch.setattr(open_models, "DNABERT2_PINNED_FILES", {remote_code.name: expected})

    open_models.verify_pinned_dnabert2_snapshot(model_dir)
    remote_code.write_text("# modified\n", encoding="utf-8")

    with pytest.raises(ValueError, match="pinned DNABERT-2 (size|SHA-256) mismatch"):
        open_models.verify_pinned_dnabert2_snapshot(model_dir)
