# Installation and Environment Setup

This guide covers setting up the development and execution environment for the Text Style & Tone Detection project, including dependencies, CPU PyTorch configuration, Git LFS downloads, and environment verification.

---

## 1. Prerequisites and Python Version

The project requires **Python 3.13** (specified as `>=3.13,<3.14` in `pyproject.toml`).

Verify your installed Python version:

```bash
python --version
```

If you manage multiple Python installations, you can use `py -3.13` on Windows or tools like `uv` or `pyenv`.

---

## 2. Package Management with uv (Recommended)

The repository uses [uv](https://docs.astral.sh/uv/) for deterministic dependency resolution via `uv.lock`.

### Installing uv

- **Windows (PowerShell):**
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
- **Linux/macOS:**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### Synchronizing the Environment

To install all standard development dependencies (defined in default groups `api`, `ui`, `data`, `dev`, `notebook`):

```bash
uv sync --frozen
```

### Dependency Groups in `pyproject.toml`

Dependencies are modularized into groups:

| Group | Key Packages | Purpose |
|---|---|---|
| `api` | `fastapi`, `uvicorn`, `torch>=2.6`, `transformers>=4.46`, `scikit-learn`, `spacy`, `en-core-web-sm` | Backend API, inference, and shared prediction logic |
| `ui` | `streamlit>=1.57.0`, `requests>=2.32` | Streamlit frontend |
| `data` | `pandas>=3.0.3` | Dataset compilation and audit |
| `dev` | `pytest>=8.3`, `httpx>=0.27`, `ruff>=0.8`, `matplotlib` | Testing, linting, formatting |
| `notebook` | `jupyterlab>=4.2`, `ipykernel>=6.29`, `seaborn` | Research notebooks and interactive analysis |
| `docs` | `mkdocs-material>=9.5` | Documentation site generation |

To install a minimal runtime environment (API + UI only):
```bash
uv sync --no-default-groups --group api --group ui
```

To include documentation tooling:
```bash
uv sync --group docs
```

---

## 3. Alternative Setup with Standard venv and pip

If `uv` is unavailable on your system, you can use standard Python virtual environments:

```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Activate on Linux/macOS
source .venv/bin/activate
```

Install packages matching `pyproject.toml`:
```bash
# Core API & CPU PyTorch
pip install --extra-index-url https://download.pytorch.org/whl/cpu torch>=2.6
pip install fastapi>=0.136.3 uvicorn>=0.48.0 joblib>=1.5.3 numpy>=2.0 pandas>=3.0.3 scikit-learn>=1.5.0 spacy>=3.8.13 transformers>=4.46
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

# UI, Dev, Notebooks
pip install streamlit>=1.57.0 requests>=2.32 pytest>=8.3 httpx>=0.27 ruff>=0.8 jupyterlab ipykernel
```

---

## 4. PyTorch CPU Wheels Configuration

The project is configured by default to use **CPU-only PyTorch** on Linux and Windows. This avoids multi-gigabyte CUDA runtime downloads while providing fast inference for small batches.

In `pyproject.toml`:
```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cpu", marker = "sys_platform != 'darwin'" }]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

On macOS, official PyPI PyTorch wheels natively support both CPU and Metal (MPS) acceleration without an explicit index override.

---

## 5. Git LFS (Large File Storage) Downloads

Binary model checkpoints (such as `best.pt`), embeddings, or serialized artifacts in Git repositories are managed through **Git LFS**.

### Installing Git LFS

1. Install the Git LFS CLI:
   - **Windows:** Download from [git-lfs.com](https://git-lfs.com/) or via `winget install GitHub.GitLFS`
   - **Ubuntu/Debian:** `sudo apt-get install git-lfs`
   - **macOS:** `brew install git-lfs`
2. Initialize Git LFS hooks:
   ```bash
   git lfs install
   ```

### Pulling Large Files

If cloned without Git LFS or if checkpoint files contain only small text pointers (starting with `version https://git-lfs.github.com/spec/v1`), fetch the actual binary payloads:

```bash
git lfs pull
```

Check the status of LFS files:
```bash
git lfs ls-files
```

---

## 6. Verifying the Environment

Run the following checks from the repository root to verify your installation:

```bash
# 1. Code linting and formatting
uv run ruff check .
uv run ruff format --check .

# 2. Test suite
uv run pytest -q

# 3. Model CLI help
uv run python -m app.prediction --help

# 4. Model selection CLI
uv run python -m scripts.select_model --list
```
