"""CSV, JSON, PNG, and descriptive error analysis for held-out evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import classification_report, confusion_matrix

from app.bundles import write_json

BUSINESS_CUES = (
    r"\b(?:meeting|report|budget|invoice|client|project|delivery|procurement|deadline)\b"
)
FORMAL_CUES = r"\b(?:hereby|pursuant|respectfully|sincerely|accordance|permission)\b"


def save_csv(frame: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    frame.to_csv(path, index=index, encoding="utf-8-sig")


def plot_confusion(counts: np.ndarray, normalized: np.ndarray, labels, path: Path) -> None:
    figure = Figure(figsize=(12, 5))
    for axis, matrix, title in zip(
        figure.subplots(1, 2),
        (counts, normalized),
        ("Counts", "Normalized by true class"),
        strict=True,
    ):
        kwargs = {"vmin": 0, "vmax": 1} if title.startswith("Normalized") else {}
        plotted = axis.imshow(matrix, cmap="Blues", **kwargs)
        axis.set(
            xticks=range(len(labels)),
            yticks=range(len(labels)),
            xticklabels=labels,
            yticklabels=labels,
            xlabel="Predicted",
            ylabel="True",
            title=title,
        )
        axis.tick_params(axis="x", rotation=45)
        for row in range(len(labels)):
            for column in range(len(labels)):
                value = matrix[row, column]
                text = f"{value:.2f}" if kwargs else str(int(value))
                axis.text(
                    column,
                    row,
                    text,
                    ha="center",
                    va="center",
                    color="white" if value > matrix.max() / 2 else "black",
                )
        figure.colorbar(plotted, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    figure.clear()


def business_formal_stats(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for source, target in (("business", "formal"), ("formal", "business")):
        support = int(frame["true_style"].eq(source).sum())
        count = int((frame["true_style"].eq(source) & frame["predicted_style"].eq(target)).sum())
        rows.append(
            {
                "true": source,
                "predicted": target,
                "count": count,
                "support": support,
                "rate": count / support if support else None,
            }
        )
    return rows


def model_analysis(result: dict, frame: pd.DataFrame, pairs: pd.DataFrame) -> str:
    lines = [
        f"## {result['run_id']}",
        "",
        f"Checkpoint: `{result['checkpoint']}`",
        "",
        f"Joint accuracy: {result['metrics']['joint_accuracy']:.4f}.",
        "",
        "### Business and formal styles",
        "",
    ]
    for row in business_formal_stats(frame):
        rate = f"{row['rate']:.2%}" if row["rate"] is not None else "n/a"
        lines.append(
            f"- {row['true']} → {row['predicted']}: {row['count']}/{row['support']} ({rate})."
        )
    swaps = frame.loc[
        (frame["true_style"].eq("business") & frame["predicted_style"].eq("formal"))
        | (frame["true_style"].eq("formal") & frame["predicted_style"].eq("business"))
    ]
    relevant = frame.loc[frame["true_style"].isin(["business", "formal"])]
    correct = relevant.loc[relevant["style_correct"]]
    lines += [
        "",
        f"Swap examples: {len(swaps)}; truncated: {int(swaps['truncated'].sum())}; "
        f"wrong with confidence >= 0.8: {int(swaps['style_confidence'].ge(0.8).sum())}.",
    ]
    if len(swaps) and len(correct):
        lines.append(
            f"Median characters: swaps={swaps['characters'].median():.1f}, "
            f"correct business/formal={correct['characters'].median():.1f}."
        )
    lines += [
        "",
        "Business subject matter and formal register can coexist. This is a hypothesis "
        "for manual review, not a measured cause. Cue flags and length/truncation strata "
        "are descriptive associations, not model feature attribution.",
        "",
        "Inspect business_formal_errors.csv and feature_strata.csv before changing labels "
        "or preprocessing. Future changes should be selected on validation data.",
        "",
        "### Most frequent directional errors",
        "",
    ]
    for row in pairs.head(10).itertuples():
        lines.append(
            f"- {row.task}: {row.true} → {row.predicted}: "
            f"{row.count}/{row.support} ({row.rate:.2%})."
        )
    if pairs.empty:
        lines.append("No classification errors on this test split.")
    lines += ["", "### Business/formal examples", ""]
    for row in swaps.head(5).itertuples():
        text = row.text.replace("\n", " ").replace("`", "'")
        lines.append(
            f"- `{row.sample_id}`: {row.true_style} → {row.predicted_style}; "
            f"confidence={row.style_confidence:.3f}. {text}"
        )
    if swaps.empty:
        lines.append("No business/formal swaps on this test split.")
    return "\n".join(lines) + "\n"


def export_model_reports(result: dict, directory: Path, labels: dict) -> None:
    directory.mkdir()
    frame = result["predictions"]
    frame["business_cue"] = frame["text"].str.contains(BUSINESS_CUES, case=False, regex=True)
    frame["formal_cue"] = frame["text"].str.contains(FORMAL_CUES, case=False, regex=True)
    identity = {
        key: value
        for key, value in result.items()
        if key not in {"predictions", "calibration_bins"}
    }
    write_json(directory / "metrics.json", identity)
    save_csv(frame, directory / "predictions.csv")
    save_csv(frame.loc[~frame["joint_correct"]], directory / "errors.csv")
    high_confidence = (~frame["style_correct"] & frame["style_confidence"].ge(0.8)) | (
        ~frame["tone_correct"] & frame["tone_confidence"].ge(0.8)
    )
    save_csv(frame.loc[high_confidence], directory / "high_confidence_errors.csv")
    swaps = (frame["true_style"].eq("business") & frame["predicted_style"].eq("formal")) | (
        frame["true_style"].eq("formal") & frame["predicted_style"].eq("business")
    )
    save_csv(frame.loc[swaps], directory / "business_formal_errors.csv")
    pairs = []
    for task, names in labels.items():
        true, predicted = frame[f"true_{task}"], frame[f"predicted_{task}"]
        report = classification_report(
            true, predicted, labels=list(names), output_dict=True, zero_division=0
        )
        write_json(directory / f"{task}_report.json", report)
        save_csv(
            pd.DataFrame.from_dict(
                {name: report[name] for name in names}, orient="index"
            ).rename_axis("class"),
            directory / f"{task}_report.csv",
            index=True,
        )
        counts = confusion_matrix(true, predicted, labels=list(names))
        support = counts.sum(axis=1, keepdims=True)
        normalized = np.divide(
            counts, support, out=np.zeros_like(counts, dtype=float), where=support > 0
        )
        for suffix, matrix in (("counts", counts), ("normalized", normalized)):
            save_csv(
                pd.DataFrame(matrix, index=names, columns=names).rename_axis("true_label"),
                directory / f"{task}_confusion_{suffix}.csv",
                index=True,
            )
        plot_confusion(counts, normalized, names, directory / f"{task}_confusion.png")
        for row, source in enumerate(names):
            for column, target in enumerate(names):
                if row != column and counts[row, column]:
                    pairs.append(
                        {
                            "task": task,
                            "true": source,
                            "predicted": target,
                            "count": int(counts[row, column]),
                            "support": int(support[row, 0]),
                            "rate": float(normalized[row, column]),
                        }
                    )
        calibration = []
        for mode in ("raw", "calibrated"):
            calibration.extend(
                {"mode": mode, **row} for row in result["calibration_bins"][f"{task}_{mode}"]
            )
        save_csv(pd.DataFrame(calibration), directory / f"{task}_calibration_bins.csv")
    pairs = pd.DataFrame(pairs, columns=["task", "true", "predicted", "count", "support", "rate"])
    pairs = pairs.sort_values("count", ascending=False, kind="stable")
    save_csv(pairs, directory / "confusion_pairs.csv")
    strata = []
    for feature in ("length_bucket", "truncated", "business_cue", "formal_cue"):
        for value, group in frame.groupby(feature, observed=True, sort=True):
            strata.append(
                {
                    "feature": feature,
                    "value": str(value),
                    "samples": len(group),
                    **{
                        f"{task}_accuracy": float(group[f"{task}_correct"].mean())
                        for task in ("style", "tone", "joint")
                    },
                }
            )
    save_csv(pd.DataFrame(strata), directory / "feature_strata.csv")
    (directory / "error_analysis.md").write_text(
        model_analysis(result, frame, pairs), encoding="utf-8"
    )


def recurring_errors(results: list[dict]) -> pd.DataFrame:
    all_predictions = pd.concat(
        [
            row["predictions"].assign(run_id=row["run_id"], architecture=row["model_name"])
            for row in results
        ],
        ignore_index=True,
    )
    records = []
    columns = [
        "sample_id",
        "text",
        "true_style",
        "true_tone",
        "style_error_architectures",
        "tone_error_architectures",
        "joint_error_architectures",
        "wrong_runs",
        "predictions",
    ]
    for sample_id, group in all_predictions.groupby("sample_id", sort=False):
        errors = {
            task: sorted(group.loc[~group[f"{task}_correct"], "architecture"].unique())
            for task in ("style", "tone", "joint")
        }
        if max(map(len, errors.values())) < 2:
            continue
        record = {key: group.iloc[0][key] for key in ("text", "true_style", "true_tone")}
        record.update(sample_id=sample_id, wrong_runs=int((~group["joint_correct"]).sum()))
        record.update(
            {f"{task}_error_architectures": ",".join(names) for task, names in errors.items()}
        )
        record["predictions"] = json.dumps(
            {
                row.run_id: {"style": row.predicted_style, "tone": row.predicted_tone}
                for row in group.itertuples()
            }
        )
        records.append(record)
    return pd.DataFrame(records, columns=columns)


def export_comparison(results: list[dict], directory: Path) -> None:
    rows, directions = [], []
    for result in results:
        row = {key: result[key] for key in ("run_id", "model_name", "checkpoint", "max_length")}
        row.update({key: result["metrics"][key] for key in ("joint_accuracy", "mean_macro_f1")})
        for task in ("style", "tone"):
            scores = result["metrics"][task]
            row.update(
                {f"{task}_{key}": scores[key] for key in ("accuracy", "macro_f1", "weighted_f1")}
            )
            for mode in ("raw", "calibrated"):
                row.update({f"{task}_{mode}_{key}": value for key, value in scores[mode].items()})
        rows.append(row)
        directions.extend(
            {"run_id": result["run_id"], "model_name": result["model_name"], **row}
            for row in business_formal_stats(result["predictions"])
        )
    save_csv(pd.DataFrame(rows), directory / "comparison.csv")
    write_json(directory / "comparison.json", rows)
    save_csv(pd.DataFrame(directions), directory / "business_formal_comparison.csv")
    recurring = recurring_errors(results)
    save_csv(recurring, directory / "recurring_errors.csv")
    lines = [
        "# Held-out test comparison",
        "",
        "All runs use the same verified test CSV and row order. "
        "Stored tokenizers, length limits, and validation temperatures are preserved.",
        "",
        "| Run | Style accuracy | Style macro F1 | "
        "Tone accuracy | Tone macro F1 | Joint accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['run_id']} | {row['style_accuracy']:.4f} | {row['style_macro_f1']:.4f} "
            f"| {row['tone_accuracy']:.4f} | {row['tone_macro_f1']:.4f} "
            f"| {row['joint_accuracy']:.4f} |"
        )
    lines += [
        "",
        f"Recurring errors across at least two distinct architectures: {len(recurring)}.",
        "Multiple runs of the same architecture do not increase the architecture count.",
        "",
        "See recurring_errors.csv for full texts and model disagreements, and "
        "business_formal_comparison.csv for directional counts divided by true-class support.",
        "",
        "Test results are descriptive. This command does not update models, temperatures, "
        "or the active-model registry.\n",
    ]
    (directory / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
