# Research Workflows, Notebooks, and Experiments

This guide explains the experimental workflow: running Jupyter research notebooks, training neural models, calibrating temperatures, and running held-out evaluations.

---

## 1. Research Notebooks Workflow

The research lifecycle is divided into three sequential notebooks located in [`notebooks/`](https://github.com/zave52/text-style-detection/tree/master/notebooks):

| Notebook | Purpose | Key Outputs |
|---|---|---|
| `01_exploratory_data_analysis.ipynb` | Dataset inspection, balance verification, length distributions, vocabulary stats | Class distributions, character/token histograms |
| `02_data_preparation_and_model_design.ipynb` | Data cleaning, conflict removal, group-aware splitting, architecture contracts | Splitting protocol, audit summary, design specs |
| `03_training_and_evaluation.ipynb` | Training run comparisons, learning curves, confusion matrices, calibration curves | Comparison tables, calibration plots |

### Configuring the Jupyter Kernel

To ensure notebooks run in the correct virtual environment:

```bash
# Register kernel
uv run python -m ipykernel install --user --name text-style-detection --display-name "Python 3.13 (text-style-detection)"

# Start JupyterLab
uv run jupyter lab
```

Inside Jupyter, select the kernel **Python 3.13 (text-style-detection)**.

### Historical Baseline Notebooks

Two root notebooks are retained as historical references:
- `modelling_02.ipynb`: TF-IDF + LinearSVC classical baseline.
- `embedding_models_03.ipynb`: spaCy `en_core_web_md` word vector averaging + SVM/RandomForest multi-output baseline.

---

## 2. Dataset Preparation (`scripts/build_dataset.py`)

Before training new models, the corpus must be built and audited:

```bash
uv run python -m scripts.build_dataset
```

What the script does:
1. Scans `dataset/{style}/{tone}/*.txt` (1000 raw files across 25 combinations).
2. Cleans text and identifies conflicts (removes 2 contradictory texts -> 998 clean samples).
3. Groups near-duplicates to prevent test leakage.
4. Performs group-aware stratified splitting: **632 train / 158 validation / 208 test**.
5. Outputs:
   - `data/raw/dataset.csv`: Compiled raw corpus.
   - `data/prepared/train.csv`, `validation.csv`, `test.csv`: Verified splits.
   - `data/prepared/manifest.json`: Fingerprints, sample counts, and split hashes.
   - `reports/corpus_audit.md` & `data_splits.md`: Audit summaries.

---

## 3. Training Neural Models (`scripts/train.py`)

The unified training script `scripts.train` is the canonical entrypoint for training all three neural architectures with early stopping, AdamW optimization, learning rate scheduling, and validation-based checkpointing.

### Command Syntax

```bash
# Train TextCNN
uv run python -m scripts.train --model textcnn --run-dir artifacts/experiments/textcnn_01 --epochs 20

# Train BiLSTM with Attention
uv run python -m scripts.train --model bilstm --run-dir artifacts/experiments/bilstm_01 --epochs 20

# Fine-tune DistilBERT
uv run python -m scripts.train --model distilbert --run-dir artifacts/experiments/distilbert_01 --epochs 5 --lr 2e-5 --head-lr 1e-3
```

### Key Training Options

| Option | Default | Description |
|---|---|---|
| `--model` | Required | Model architecture: `textcnn`, `bilstm`, or `distilbert` |
| `--run-dir` | Required | Destination directory for model bundle and metrics |
| `--epochs` | 20 | Maximum training epochs |
| `--batch-size` | 32 | Batch size for training and validation loaders across all models |
| `--max-length` | 128 | Maximum sequence length limit for tokenization |
| `--lr` | Model default (1e-3 for CNN/BiLSTM, 2e-5 for DistilBERT) | Encoder learning rate |
| `--head-lr` | 1e-4 | Dedicated learning rate for classification heads (applies to DistilBERT only) |
| `--weight-decay` | 1e-2 | AdamW weight decay |
| `--max-grad-norm` | 1.0 | Gradient clipping threshold |
| `--patience` | 5 | Early stopping patience (epochs without validation improvement) |
| `--style-weight` | 1.0 | Loss multiplier for style cross-entropy |
| `--tone-weight` | 1.0 | Loss multiplier for tone cross-entropy |
| `--device` | `TSD_DEVICE` (default `cpu`) | Training device (`cpu`, `cuda`, `auto`) |
| `--evaluate-test` | False (off) | Run one-shot test evaluation after training |
| `--config` | None | Path to JSON training configuration file |

---

## 4. Temperature Calibration (`scripts/calibrate.py`)

Post-training calibration fits temperature scaling parameters $T_{\text{style}}$ and $T_{\text{tone}}$ on validation logits via bisection search on inverse temperature $1/T$ to minimize validation negative log-likelihood (NLL).

Modern runs created via `scripts.train` perform calibration automatically upon training completion. For unsealed experiment directories, run:

```bash
uv run python -m scripts.calibrate --checkpoint artifacts/experiments/textcnn_01
```

If an incomplete calibration file exists, add `--force` to recompute:
```bash
uv run python -m scripts.calibrate --checkpoint artifacts/experiments/textcnn_01 --force
```

---

## 5. Held-Out Evaluation (`scripts/evaluate.py`)

Evaluate calibrated checkpoints on the held-out test split (`data/prepared/test.csv`).

Note that `--output` must not exist prior to running evaluation. By default, `--require-all` is active and expects checkpoints for all three architectures (`textcnn`, `bilstm`, `distilbert`). When evaluating a subset of models, pass `--no-require-all`:

```bash
# Evaluate all three architectures
uv run python -m scripts.evaluate \
  --checkpoints artifacts/experiments/distilbert_01 artifacts/experiments/textcnn_01 artifacts/experiments/bilstm_01 \
  --output reports/benchmark

# Or evaluate specific models with --no-require-all
uv run python -m scripts.evaluate \
  --checkpoints artifacts/experiments/distilbert_01 artifacts/experiments/textcnn_01 \
  --no-require-all \
  --output reports/benchmark_subset
```

### Generated Reports Layout

The script generates a comprehensive report directory:

- Root of `--output`:
  - `comparison.csv`: Summary table comparing models across accuracy, macro F1, weighted F1, joint accuracy, and ECE.
  - `comparison.json`, `comparison.md`: Machine-readable and formatted summaries.
  - `recurring_errors.csv`: Analysis of test samples consistently misclassified across models.
- Per-model subdirectories (`<output>/<model>/`):
  - `metrics.json`: Detailed numeric metrics for both style and tone tasks.
  - `predictions.csv`: Row-by-row predicted and true labels with calibrated probabilities.
  - `errors.csv`, `high_confidence_errors.csv`, `business_formal_errors.csv`: Filtered error slices.
  - `style_report.csv`, `tone_report.csv`: Per-class precision, recall, and F1 scores.
  - `style_confusion_counts.csv`, `style_confusion_normalized.csv`, `style_confusion.png`: Confusion matrices.
  - `tone_confusion_counts.csv`, `tone_confusion_normalized.csv`, `tone_confusion.png`: Confusion matrices.
  - `confusion_pairs.csv`: Most frequent class confusions sorted by count.
  - `style_calibration_bins.csv`, `tone_calibration_bins.csv`: Reliability diagram bin data.
  - `feature_strata.csv`: Performance segmented by text length and linguistic cues.
  - `error_analysis.md`: Detailed qualitative error report.
