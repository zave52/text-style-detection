"""Persistent selection of a verified, locally saved neural model."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from app.bundles import bundle_directory, read_json, sha256, verify_bundle, write_json
from app.config import ARTIFACTS_DIR

DEFAULT_REGISTRY = ARTIFACTS_DIR / "active_model.json"


class ActiveModelRegistry:
    def __init__(self, path: str | Path = DEFAULT_REGISTRY) -> None:
        self.path = Path(path).resolve()

    def read(self) -> dict:
        if not self.path.exists():
            return {
                "version": 1,
                "active": None,
                "models": {},
            }

        registry = read_json(self.path)

        if registry.get("version") != 1 or not isinstance(registry.get("models"), dict):
            raise ValueError("Unsupported active-model registry format.")

        return registry

    def select(
        self,
        checkpoint: str | Path,
        name: str | None = None,
    ) -> dict:
        from app.prediction import NeuralPredictor

        root = bundle_directory(checkpoint)
        manifest = verify_bundle(root)

        # Validate tokenizer, output dimensions, and state_dict before switching.
        NeuralPredictor(root, device="cpu")

        name = name or manifest["model_name"]

        if not isinstance(name, str) or not name.strip():
            raise ValueError("Registry model name must be non-empty.")

        try:
            location = os.path.relpath(root, self.path.parent)
        except ValueError:
            # Different Windows drives.
            location = str(root)

        registry = self.read()
        registry["models"][name] = {
            "directory": Path(location).as_posix(),
            "model_name": manifest["model_name"],
            "weights_sha256": manifest["weights_sha256"],
            "bundle_sha256": sha256(root / "bundle.json"),
            "selected_at": datetime.now(UTC).isoformat(),
        }
        registry["active"] = name

        write_json(self.path, registry)
        return registry["models"][name]

    def resolve(self) -> Path:
        registry = self.read()
        active = registry.get("active")

        if active not in registry["models"]:
            raise FileNotFoundError(
                "No active model. Run python -m scripts.select_model first."
            )

        entry = registry["models"][active]
        root = (self.path.parent / entry["directory"]).resolve()

        manifest = verify_bundle(
            root,
            expected_sha256=entry["bundle_sha256"],
        )

        if (
            manifest["weights_sha256"] != entry["weights_sha256"]
            or manifest["model_name"] != entry["model_name"]
        ):
            raise ValueError("Selected model does not match its registry entry.")

        return root
