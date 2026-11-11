from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.experiments import TrainConfig, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON settings; CLI flags override its values.")
    parser.add_argument("--model", dest="model_name", choices=("textcnn", "bilstm", "distilbert"))
    parser.add_argument("--run-dir")
    for flag in ("epochs", "batch-size", "max-length", "patience", "lr-patience", "seed"):
        parser.add_argument(f"--{flag}", type=int)
    for flag in (
        "weight-decay",
        "max-grad-norm",
        "style-weight",
        "tone-weight",
        "min-delta",
        "lr-factor",
    ):
        parser.add_argument(f"--{flag}", type=float)
    parser.add_argument("--lr", dest="learning_rate", type=float)
    parser.add_argument("--head-lr", dest="head_learning_rate", type=float)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--evaluate-test", action=argparse.BooleanOptionalAction, default=None)
    args = vars(parser.parse_args())
    source = args.pop("config")
    saved = json.loads(source.read_text(encoding="utf-8")) if source else {}
    settings = dict(saved.get("settings", saved))
    settings.update({key: value for key, value in args.items() if value is not None})
    run_experiment(TrainConfig(**settings))


if __name__ == "__main__":
    main()
