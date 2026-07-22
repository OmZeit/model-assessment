"""Smoke an installed wheel from outside the source tree; invoked directly by CI."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from importlib import metadata, resources
from pathlib import Path


EXPECTED_RESOURCES = [
    "assets/assayready.css",
    "data_core/bpe_vocab/tokenizer.json",
    "examples/prediction_audit.csv",
    "examples/prediction_audit.json",
    "examples/prediction_candidates.csv",
    "examples/public_dream_promoter_audit.json",
    "examples/public_dream_promoter_candidates.csv",
    "examples/public_dream_promoter_predictions.csv",
]


def run(command: list[str], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    import model_assessment
    from model_assessment.data_core.tokenizer import DnaTokenizer
    from model_assessment.ui import create_app

    source_package = Path(__file__).resolve().parents[1] / "model_assessment"
    installed_package = Path(model_assessment.__file__).resolve().parent
    if installed_package == source_package.resolve():
        raise RuntimeError("Smoke test imported the source tree instead of the installed wheel.")
    if metadata.version("model-assessment") != model_assessment.__version__:
        raise RuntimeError("Wheel metadata and package versions differ.")

    package_root = resources.files("model_assessment")
    missing = [name for name in EXPECTED_RESOURCES if not package_root.joinpath(name).is_file()]
    if missing:
        raise RuntimeError(f"Wheel is missing package resources: {missing}")

    tokenizer = DnaTokenizer()
    if tokenizer.vocab_size <= 5:
        raise RuntimeError("Bundled tokenizer did not load a usable vocabulary.")
    app = create_app()
    if app.title != "AssayReady":
        raise RuntimeError("Bundled UI did not initialize.")

    with tempfile.TemporaryDirectory(prefix="assayready-wheel-smoke-") as temp_name:
        temp_dir = Path(temp_name)
        run([sys.executable, "-m", "model_assessment.cli", "doctor"], cwd=temp_dir)
        with resources.as_file(package_root.joinpath("examples")) as example_dir:
            run(
                [
                    sys.executable,
                    "-m",
                    "model_assessment.cli",
                    "audit-predictions",
                    "--config",
                    str(example_dir / "prediction_audit.json"),
                ],
                cwd=temp_dir,
            )
        output_dir = temp_dir / "outputs" / "assayready" / "example_prediction_audit"
        report_path = output_dir / "prediction_audit_report.json"
        if not report_path.is_file():
            raise RuntimeError(f"Installed-wheel audit did not write {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not report.get("verdict"):
            raise RuntimeError("Installed-wheel audit report has no verdict.")

    print(f"Installed wheel smoke passed: {installed_package}")


if __name__ == "__main__":
    main()
