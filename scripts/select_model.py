"""Select a calibrated checkpoint for subsequent local predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.registry import DEFAULT_REGISTRY, ActiveModelRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Experiment directory or its best.pt.",
    )
    parser.add_argument(
        "--name",
        help="Registry alias; defaults to the architecture name.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Show registered models and exit.",
    )
    args = parser.parse_args()

    registry = ActiveModelRegistry(args.registry)

    if args.list:
        result = registry.read()
    else:
        if args.checkpoint is None:
            parser.error("--checkpoint is required when selecting a model.")

        result = registry.select(args.checkpoint, args.name)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
