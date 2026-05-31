# Dataset Overview

## Summary

The dataset contains **1 000 English text samples** with two independent categorical labels: **style** and **tone**. The dataset is perfectly balanced across all classes.

| Parameter | Value |
|-----------|-------|
| Total samples | 1 000 |
| Styles | 5 |
| Tones | 5 |
| Unique combinations | 25 (5 × 5) |
| Samples per combination | 40 |
| Samples per style | 200 |
| Samples per tone | 200 |

---

## Label Taxonomy

### Writing Styles

| Style | Description |
|-------|-------------|
| `academic` | Scholarly, research-oriented writing with formal vocabulary and citations |
| `business` | Professional corporate communication — emails, reports, memos |
| `formal` | Official language following established conventions without domain-specific jargon |
| `informal` | Casual, conversational writing — chats, personal messages, social media |
| `literary` | Creative, artistic prose with figurative language and narrative elements |

### Emotional Tones

| Tone | Description |
|------|-------------|
| `aggressive` | Hostile, confrontational, or combative language |
| `friendly` | Warm, supportive, and approachable communication |
| `neutral` | Objective, factual, and emotionally detached |
| `sarcastic` | Ironic statements where the intended meaning differs from the literal meaning |
| `urgent` | Time-sensitive, pressing, and action-demanding language |

---

## File Organization

### Raw Text Files

The raw corpus is organized in a two-level directory hierarchy:

```
dataset/
├── academic/
│   ├── aggressive/
│   │   ├── 001.txt
│   │   ├── 002.txt
│   │   └── ...          # 40 files per combination
│   ├── friendly/
│   ├── neutral/
│   ├── sarcastic/
│   └── urgent/
├── business/
│   ├── aggressive/
│   ├── friendly/
│   └── ...
├── formal/
├── informal/
└── literary/
```

Each `.txt` file contains a single text sample written in a specific style–tone combination.

### Compiled CSV

The script `scripts/build_dataset.py` compiles all raw text files into a single CSV:

```
data/dataset.csv
```

| Column | Type | Description |
|--------|------|-------------|
| `text` | `str` | Original raw text |
| `clean_text` | `str` | Preprocessed text (lemmatized, cleaned) |
| `style` | `str` | One of 5 style labels |
| `tone` | `str` | One of 5 tone labels |

---

## Class Distribution

The dataset is **perfectly balanced**: every style has exactly 200 samples, and every tone has exactly 200 samples. Every style × tone combination has exactly 40 samples.

### Style Distribution

| Style | Count | Share |
|-------|-------|-------|
| academic | 200 | 20% |
| business | 200 | 20% |
| formal | 200 | 20% |
| informal | 200 | 20% |
| literary | 200 | 20% |

### Tone Distribution

| Tone | Count | Share |
|------|-------|-------|
| aggressive | 200 | 20% |
| friendly | 200 | 20% |
| neutral | 200 | 20% |
| sarcastic | 200 | 20% |
| urgent | 200 | 20% |

### Cross-Tabulation (Style × Tone)

|  | aggressive | friendly | neutral | sarcastic | urgent |
|--|-----------|----------|---------|-----------|--------|
| **academic** | 40 | 40 | 40 | 40 | 40 |
| **business** | 40 | 40 | 40 | 40 | 40 |
| **formal** | 40 | 40 | 40 | 40 | 40 |
| **informal** | 40 | 40 | 40 | 40 | 40 |
| **literary** | 40 | 40 | 40 | 40 | 40 |

---

## Building the Dataset

To regenerate `data/dataset.csv` from the raw text files:

```bash
cd scripts
python build_dataset.py
```

The script:

1. Iterates over all `dataset/{style}/{tone}/*.txt` files
2. Reads and preprocesses each text (see [Preprocessing](preprocessing.md))
3. Compiles everything into a DataFrame
4. Shuffles with `random_state=42`
5. Saves to `data/dataset.csv`

!!! note "Reproducibility"
    The shuffle uses a fixed random seed (`random_state=42`), so re-running the script produces an identical CSV.
