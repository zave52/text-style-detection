# Troubleshooting, Known Issues, and System Limitations

This document catalogs common issues encountered during installation, notebook execution, model serving, and validation, along with their solutions and design limitations.

---

## 1. Missing Packages and Dependency Groups

### Problem: `ModuleNotFoundError: No module named 'mkdocs'` or similar
- **Cause:** Dependencies in this project are grouped into modular sets in `pyproject.toml`. Standard `uv sync` installs default groups, but optional groups (such as `docs`) require explicit activation.
- **Solution:**
  ```bash
  # For documentation tools
  uv sync --group docs

  # For full environment
  uv sync --group docs --group dev --group notebook
  ```

### Problem: `spacy` model `en_core_web_sm` is missing
- **Cause:** In standard pip environments, spaCy models must be installed separately.
- **Solution:**
  ```bash
  pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
  ```
  Or via `uv sync --frozen` which installs the wheel pinned in `pyproject.toml`.

---

## 2. Jupyter Notebook Kernel Issues

### Problem: Imports fail inside Jupyter notebooks despite installing dependencies
- **Cause:** Jupyter may default to the system Python kernel rather than the project virtual environment.
- **Solution:**
  Register and explicitly choose the project kernel:
  ```bash
  uv run python -m ipykernel install --user --name text-style-detection --display-name "Python 3.13 (text-style-detection)"
  ```
  In JupyterLab or VS Code, click the kernel selector in the top-right corner and choose **Python 3.13 (text-style-detection)**.

---

## 3. CPU Execution Considerations

### Resource Usage and Batch Limits
- PyTorch is configured by default with CPU wheels.
- For batch inference, keep batch sizes between 1 and 64 to avoid memory pressure and latency spikes.
- By default, PyTorch utilizes available CPU cores for intra-op parallelism.

### Using GPU Acceleration (Optional)
If a CUDA-capable GPU is available and you want to use it for training:
```bash
uv run python -m scripts.train --model distilbert --device cuda
```
Ensure a CUDA-enabled PyTorch wheel is installed in your virtual environment.

---

## 4. Checkpoint Integrity and Bundle Validation

### Problem: `ValueError: Checkpoint directory ... is missing bundle.json` or `Checksum mismatch`
- **Cause:** Model loading requires a complete, sealed experiment directory. Copying `best.pt` alone without its accompanying JSON descriptors will cause loading to fail.
- **Required bundle files:**
  - `best.pt`: Serialized PyTorch state dictionary.
  - `bundle.json`: File list and SHA-256 hashes.
  - `calibration.json`: Fitted temperature parameters.
  - `labels.json`: Canonical ordering of style and tone classes.
  - `model_config.json`: Architecture hyperparameters.
  - `training_metadata.json`: Training timestamps, device, epochs.
  - `data_manifest.json`: Fingerprint of dataset splits used.
  - `vocabulary.json` (for TextCNN / BiLSTM) or `tokenizer/` + `config.json` (for DistilBERT).
- **Solution:**
  Ensure the entire directory under `artifacts/experiments/<model_name>/` is transferred and intact. Never manually edit sealed JSON files.

### Problem: Incomplete calibration found
- **Cause:** Calibration was interrupted or unsealed.
- **Solution:**
  Run `scripts.calibrate` with `--force` to recompute temperatures:
  ```bash
  uv run python -m scripts.calibrate --checkpoint artifacts/experiments/<run_name> --force
  ```

---

## 5. Windows-Specific Quirks

### Problem: `UnicodeDecodeError: 'charmap' codec can't decode byte ...`
- **Cause:** On Windows, Python functions like `open()` and `pd.read_csv()` historically default to the system locale encoding (such as `cp1252` or `windows-1251`) instead of UTF-8 when `encoding` is omitted.
- **Solution:**
  Always specify `encoding="utf-8"` when reading text files and CSVs:
  ```python
  pd.read_csv("path/to/file.csv", encoding="utf-8")
  ```

### Problem: Docker bind mount path differences
- **Cause:** Windows backslashes `\` inside environment variables passed to Linux containers (e.g., `TSD_CHECKPOINT=artifacts\experiments\distilbert_01`).
- **Solution:**
  Inside Docker containers, all paths must use Linux forward slashes `/app/artifacts/...`. Use relative paths or let `TSD_REGISTRY=/app/artifacts/active_model.json` resolve paths automatically.

---

## 6. Model and Scope Limitations

- **Corpus Scale and Balance:** The dataset is a curated, balanced corpus of 1000 messages (200 per style, 200 per tone). In production scenarios, real-world class distributions, message lengths, and vocabularies may differ.
- **English Language Constraint:** The models are trained strictly on English texts. The `unfamiliar_input` warning checks for non-Latin character ratios and out-of-vocabulary tokens, which is an operational heuristic rather than a full language identification tool.
- **Context Absence:** Classification evaluates each submitted message in isolation. Sarcasm, communicative intent, quotations, and colloquial brevity often depend on external dialogue or situational context unavailable to the model.
- **Class Boundary Overlap:** Certain writing styles (particularly `business` and `formal`) naturally exhibit structural and linguistic overlap. The model learns boundaries defined by the dataset annotations.
- **Closed-World Assumption:** The system assigns exactly one style label and one tone label from the predetermined taxonomy. There is no open-ended "unknown" class; uncertainty is signaled via structured warnings (`uncertain_prediction`) and confidence scores.
- **Sequence Truncation:** Tokens beyond the sequence length limit (`max_length`) are truncated and do not contribute to classification. High model confidence does not prevent classification errors on out-of-domain text.
