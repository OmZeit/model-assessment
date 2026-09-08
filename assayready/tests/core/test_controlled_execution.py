import csv
from pathlib import Path

import pytest

from model_assessment.controlled_execution import (
    ContainerExecutionSpec,
    build_runtime_command,
    validate_prediction_output,
)


DIGEST_IMAGE = "registry.example/assay/model@sha256:" + "a" * 64


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_controlled_command_is_immutable_networkless_and_non_shell(tmp_path: Path) -> None:
    locked = tmp_path / "locked.csv"
    _write_csv(locked, [{"sequence_id": "a", "sequence": "ACGT"}])
    output = tmp_path / "output"
    output.mkdir()
    spec = ContainerExecutionSpec(
        image=DIGEST_IMAGE,
        input_file=locked,
        output_dir=output,
        allowed_output_root=tmp_path,
        gpu_device="0",
    )
    command = build_runtime_command(spec)
    assert command[:3] == ["docker", "run", "--rm"]
    assert "--network=none" in command
    assert "--read-only" in command
    assert "--tmpfs=/tmp:rw,noexec,nosuid,size=256m" in command
    assert "--cap-drop=ALL" in command
    assert "--gpus=device=0" in command
    assert DIGEST_IMAGE in command
    assert (
        f"type=bind,source={locked},target=/assayready/input/{locked.name},readonly"
        in command
    )
    assert f"type=bind,source={locked.parent},target=/assayready/input,readonly" not in command


def test_mutable_image_tag_is_rejected(tmp_path: Path) -> None:
    locked = tmp_path / "locked.csv"
    _write_csv(locked, [{"sequence_id": "a", "sequence": "ACGT"}])
    spec = ContainerExecutionSpec(
        image="example/model:latest",
        input_file=locked,
        output_dir=tmp_path,
        allowed_output_root=tmp_path,
    )
    with pytest.raises(ValueError, match="immutable"):
        spec.validate()


def test_placeholder_image_digest_is_rejected_with_actionable_message(tmp_path: Path) -> None:
    locked = tmp_path / "locked.csv"
    _write_csv(locked, [{"sequence_id": "a", "sequence": "ACGT"}])
    spec = ContainerExecutionSpec(
        image="registry.example.org/team/assayready-dnabert2@sha256:replace-with-registry-digest",
        input_file=locked,
        output_dir=tmp_path,
        allowed_output_root=tmp_path,
    )
    with pytest.raises(ValueError, match="built and pushed container digest"):
        spec.validate()


def test_gpu_device_must_be_a_single_index_or_all(tmp_path: Path) -> None:
    locked = tmp_path / "locked.csv"
    _write_csv(locked, [{"sequence_id": "a", "sequence": "ACGT"}])
    spec = ContainerExecutionSpec(
        image=DIGEST_IMAGE,
        input_file=locked,
        output_dir=tmp_path,
        allowed_output_root=tmp_path,
        gpu_device="0,1",
    )
    with pytest.raises(ValueError, match="gpu_device"):
        spec.validate()


def test_prediction_output_must_match_locked_identifiers(tmp_path: Path) -> None:
    locked = tmp_path / "locked.csv"
    output = tmp_path / "predictions.csv"
    _write_csv(locked, [{"sequence_id": "a"}, {"sequence_id": "b"}])
    _write_csv(output, [{"sequence_id": "a", "prediction": 0.5}, {"sequence_id": "c", "prediction": 0.7}])
    spec = ContainerExecutionSpec(
        image=DIGEST_IMAGE,
        input_file=locked,
        output_dir=tmp_path,
        allowed_output_root=tmp_path,
    )
    with pytest.raises(ValueError, match="identifiers do not match"):
        validate_prediction_output(spec, output)


def test_prediction_output_validation_records_checksum(tmp_path: Path) -> None:
    locked = tmp_path / "locked.csv"
    output = tmp_path / "predictions.csv"
    _write_csv(locked, [{"sequence_id": "a"}, {"sequence_id": "b"}])
    _write_csv(output, [{"sequence_id": "b", "prediction": 0.7}, {"sequence_id": "a", "prediction": 0.5}])
    spec = ContainerExecutionSpec(
        image=DIGEST_IMAGE,
        input_file=locked,
        output_dir=tmp_path,
        allowed_output_root=tmp_path,
    )
    result = validate_prediction_output(spec, output)
    assert result["id_set_match"] is True
    assert len(result["output_sha256"]) == 64


def test_output_dir_must_remain_within_controlled_root(tmp_path: Path) -> None:
    locked = tmp_path / "locked.csv"
    _write_csv(locked, [{"sequence_id": "a", "sequence": "ACGT"}])
    spec = ContainerExecutionSpec(
        image=DIGEST_IMAGE,
        input_file=locked,
        output_dir=tmp_path / "outside",
        allowed_output_root=tmp_path / "authorized",
    )

    with pytest.raises(ValueError, match="controlled output root"):
        spec.validate()


@pytest.mark.parametrize("memory", ["", "0g", "4", "4gb", "1.5g", "4g --privileged"])
def test_memory_limit_has_strict_container_format(tmp_path: Path, memory: str) -> None:
    locked = tmp_path / "locked.csv"
    _write_csv(locked, [{"sequence_id": "a", "sequence": "ACGT"}])
    spec = ContainerExecutionSpec(
        image=DIGEST_IMAGE,
        input_file=locked,
        output_dir=tmp_path,
        allowed_output_root=tmp_path,
        memory=memory,
    )

    with pytest.raises(ValueError, match="memory"):
        spec.validate()


def test_mapping_uses_spec_local_outputs_as_default_root(tmp_path: Path) -> None:
    locked = tmp_path / "locked.csv"
    _write_csv(locked, [{"sequence_id": "a", "sequence": "ACGT"}])
    accepted = ContainerExecutionSpec.from_mapping(
        {"image": DIGEST_IMAGE, "input_file": locked.name, "output_dir": "outputs/run"},
        base_dir=tmp_path,
    )
    accepted.validate()

    escaped = ContainerExecutionSpec.from_mapping(
        {"image": DIGEST_IMAGE, "input_file": locked.name, "output_dir": "../outside"},
        base_dir=tmp_path,
    )
    with pytest.raises(ValueError, match="controlled output root"):
        escaped.validate()


def test_configured_controlled_output_root_must_be_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ASSAYREADY_CONTROLLED_OUTPUT_ROOT", "relative-output")

    with pytest.raises(RuntimeError, match="absolute"):
        ContainerExecutionSpec.from_mapping(
            {"image": DIGEST_IMAGE, "input_file": "locked.csv", "output_dir": "outputs/run"},
            base_dir=tmp_path,
        )
