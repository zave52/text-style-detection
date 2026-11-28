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
   - `data/dataset.csv`: Compiled raw corpus.
   - `data/prepared/train.csv`, `validation.csv`, `test.csv`: Verified splits.
   - `data/prepared/manifest.json`: Fingerprints, sample counts, and split hashes.
   - `reports/corpus_audit.md` & `data_splits.md`: Audit summaries.

---

## 3. Training Neural Models (`scripts/train.py`)

The unified training script `scripts.train` supports all three neural architectures with early stopping, AdamW optimization, learning rate scheduling, and validation-based checkpointing.

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
| `--run-dir` | Auto | Destination directory for model bundle and metrics |
| `--epochs` | 20 | Maximum training epochs |
| `--batch-size` | 32 (16 for DistilBERT) | Batch size for training and validation loaders |
| `--lr` | 1e-3 (2e-5 for DistilBERT) | Encoder learning rate |
| `--head-lr` | None | Dedicated learning rate for classification heads |
| `--weight-decay` | 1e-2 | AdamW weight decay |
| `--max-grad-norm` | 1.0 | Gradient clipping threshold |
| `--patience` | 5 | Early stopping patience (epochs without validation improvement) |
| `--style-weight` | 1.0 | Loss multiplier for style cross-entropy |
| `--tone-weight` | 1.0 | Loss multiplier for tone cross-entropy |
| `--device` | auto | Training device (`cpu`, `cuda`, `auto`) |
| `--config` | None | Path to JSON training configuration file |

Standalone entrypoints are also available:
- `python -m scripts.train_textcnn`
- `python -m scripts.train_bilstm`
- `python -m scripts.train_distilbert`

---

## 4. Temperature Calibration (`scripts/calibrate.py`)

Post-training calibration fits temperature scaling parameters $T_{\text{style}}$ and $T_{\text{tone}}$ on validation logits to minimize negative log-likelihood (NLL) and expected calibration error (ECE).

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

Evaluate one or more calibrated checkpoints on the held-out test split (`data/prepared/test.csv`):

```bash
uv run python -m scripts.evaluate \
  --checkpoints artifacts/experiments/distilbert_01 artifacts/experiments/textcnn_01 \
  --output reports/benchmark
```

### Generated Reports in Output Directory

- `comparison.csv`: Summary table comparing models across accuracy, macro F1, weighted F1, joint accuracy, and ECE.
- `{model}_classification_report.csv`: Detailed per-class precision, recall, and F1 scores.
- `{model}_confusion_matrix.csv`: Normalized and raw confusion matrices for style and tone.
- `predictions.csv`: Row-by-row predicted vs ground truth labels and calibrated probabilities.
