"""Self-contained model bundles and SHA-256 integrity checks."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import torch

from app.calibration import temperatures_from
from app.tokenization import Vocabulary

CORE_FILES = {
    "best.pt",
    "calibration.json",
    "labels.json",
    "model_config.json",
    "training_metadata.json",
    "training_config.json",
    "data_manifest.json",
    "history.json",
    "summary.json",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)

        try:
            json.dump(
                value,
                stream,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise

    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def bundle_directory(path: str | Path) -> Path:
    path = Path(path).resolve()

    if path.is_file():
        if path.name != "best.pt":
            raise ValueError("A calibrated bundle must use best.pt.")
        path = path.parent

    return path


def _local_file(root: Path, name: str) -> Path:
    relative = Path(name)

    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError(f"Invalid bundle path: {name}")

    path = root / relative

    if not path.resolve().is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or invalid bundle file: {name}")

    return path


def _required_files(root: Path, model_name: str) -> set[str]:
    if model_name not in {"textcnn", "bilstm", "distilbert"}:
        raise ValueError("Unknown bundle architecture.")

    required = set(CORE_FILES)

    if model_name == "distilbert":
        required.update(
            {
                "config.json",
                "tokenizer/tokenizer_config.json",
            }
        )

        tokenizer_files = {
            path.relative_to(root).as_posix()
            for path in (root / "tokenizer").rglob("*")
            if path.is_file()
        }

        if not {"tokenizer/tokenizer.json", "tokenizer/vocab.txt"} & tokenizer_files:
            raise ValueError("Missing local transformer tokenizer vocabulary.")

        required.update(tokenizer_files)
    else:
        required.add("vocabulary.json")

    return required


def recover_incomplete_bundle(directory: str | Path) -> Path:
    """Explicit --force recovery; completed bundles must never be overwritten."""
    root = bundle_directory(directory)
    if (root / "bundle.json").exists():
        raise FileExistsError("This experiment is already calibrated and sealed.")

    checkpoint = torch.load(root / "best.pt", map_location="cpu", weights_only=True)
    if "calibration" in checkpoint:
        del checkpoint["calibration"]
        temporary = root / "best.pt.recovery.tmp"
        try:
            torch.save(checkpoint, temporary)
            temporary.replace(root / "best.pt")
        finally:
            temporary.unlink(missing_ok=True)

    # Remove this last: another interrupted recovery can safely retry --force.
    (root / "calibration.json").unlink(missing_ok=True)
    return root


def save_calibrated_bundle(directory: str | Path, calibration: dict) -> Path:
    root = bundle_directory(directory)

    if (root / "bundle.json").exists():
        raise FileExistsError("This experiment is already sealed; choose a new run directory.")

    temperatures_from(calibration)

    # Check training provenance before changing the original checkpoint.
    for name in (
        "training_config.json",
        "data_manifest.json",
        "history.json",
        "summary.json",
    ):
        read_json(root / name)

    checkpoint = torch.load(
        root / "best.pt",
        map_location="cpu",
        weights_only=True,
    )
    checkpoint["calibration"] = calibration
    name = checkpoint["model_name"]

    if name != "distilbert":
        Vocabulary(
            checkpoint["vocab_tokens"],
            checkpoint["max_length"],
        ).save(root)

    config = {key: checkpoint[key] for key in ("model_name", "model_kwargs", "max_length")}
    config["tokenizer_kind"] = checkpoint.get("tokenizer_kind", "vocabulary")

    write_json(root / "model_config.json", config)
    write_json(root / "calibration.json", calibration)
    write_json(
        root / "labels.json",
        {key: checkpoint[key] for key in ("styles", "tones")},
    )
    write_json(
        root / "training_metadata.json",
        checkpoint.get("metadata", {}),
    )

    required = _required_files(root, name)
    temporary = root / "best.pt.tmp"

    try:
        torch.save(checkpoint, temporary)
        temporary.replace(root / "best.pt")
    finally:
        temporary.unlink(missing_ok=True)

    for optional in ("optimizer.json", "model_kwargs.json", "test_metrics.json"):
        if (root / optional).is_file():
            required.add(optional)

    checksums = {name: sha256(_local_file(root, name)) for name in sorted(required)}

    write_json(
        root / "bundle.json",
        {
            "version": 1,
            "model_name": checkpoint["model_name"],
            "checkpoint": "best.pt",
            "weights_sha256": checksums["best.pt"],
            "dataset_manifest_sha256": checksums["data_manifest.json"],
            "files": checksums,
        },
    )

    return root


def verify_bundle(
    directory: str | Path,
    expected_sha256: str | None = None,
) -> dict:
    root = bundle_directory(directory)
    manifest_path = root / "bundle.json"

    if expected_sha256 is not None and sha256(manifest_path) != expected_sha256:
        raise ValueError("Selected bundle manifest checksum mismatch.")

    manifest = read_json(manifest_path)

    if manifest.get("version") != 1 or manifest.get("checkpoint") != "best.pt":
        raise ValueError("Unsupported model bundle format.")

    files = manifest["files"]

    if not _required_files(root, manifest["model_name"]).issubset(files):
        raise ValueError("Bundle manifest omits required files.")

    for name, expected in files.items():
        if sha256(_local_file(root, name)) != expected:
            raise ValueError(f"SHA-256 checksum mismatch: {name}")

    if manifest["weights_sha256"] != files["best.pt"]:
        raise ValueError("Weight checksum does not match the bundle manifest.")

    if manifest["dataset_manifest_sha256"] != files["data_manifest.json"]:
        raise ValueError("Dataset manifest checksum does not match the bundle manifest.")

    temperatures_from(read_json(root / "calibration.json"))
    return manifest
