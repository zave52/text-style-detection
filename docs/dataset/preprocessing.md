# Text Preprocessing

The preprocessing pipeline is shared between the dataset building step (`scripts/build_dataset.py`) and the inference backend (`app/ml.py`). This ensures consistency between training and serving.

---

## Pipeline Steps

The preprocessing transforms raw text into a clean, normalized form suitable for feature extraction:

```mermaid
flowchart LR
    A[Raw Text] --> B[URL Removal]
    B --> C[spaCy Tokenization]
    C --> D[Lemmatization]
    D --> E[Lowercasing]
    E --> F[Space Normalization]
    F --> G[Clean Text]
```

### Step 1 — URL Removal

All URLs are stripped using a regular expression:

```python
text = re.sub(r"http\S+|www\S+", "", text)
```

This removes any token starting with `http` or `www` (including the rest of the token up to the next whitespace).

### Step 2 — spaCy Tokenization & Lemmatization

The text is processed through the **spaCy** `en_core_web_sm` pipeline, which provides:

- Tokenization (splitting into words, punctuation, etc.)
- Part-of-speech tagging
- Lemmatization (reducing words to their base form)

```python
doc = nlp(text)
```

### Step 3 — Token Filtering & Normalization

Each token is processed according to these rules:

| Token Type | Action | Example |
|-----------|--------|---------|
| Whitespace (`token.is_space`) | **Skip** | `\n`, `\t` → removed |
| Punctuation (`token.is_punct`) | **Keep as-is** | `.`, `,`, `!` → preserved |
| Other tokens | **Lemmatize + lowercase** | `"running"` → `"run"`, `"Better"` → `"good"` |

```python
for token in doc:
    if token.is_space:
        continue
    if token.is_punct:
        clean_tokens.append(token.text)
    else:
        lemma = token.lemma_.strip()
        if lemma:
            clean_tokens.append(lemma.lower())
```

### Step 4 — Space Normalization

Multiple consecutive spaces are collapsed into a single space, and leading/trailing whitespace is removed:

```python
clean_text = " ".join(clean_tokens)
clean_text = re.sub(r"\s+", " ", clean_text).strip()
```

---

## Example

**Input:**

> "The researchers have been aggressively investigating the new hypothesis since last Monday. Visit http://example.com for details."

**Output:**

> `the researcher have be aggressively investigate the new hypothesis since last Monday . visit for detail .`

Key transformations:

- `researchers` → `researcher` (lemmatization)
- `have been` → `have be` (lemmatization)
- `investigating` → `investigate` (lemmatization)
- `http://example.com` → removed (URL removal)
- `details` → `detail` (lemmatization)
- Punctuation `.` preserved

---

## Design Decisions

!!! info "Why preserve punctuation?"
    Punctuation carries stylistic and tonal signals. For example, exclamation marks (`!`) may indicate urgency or aggression, while ellipses (`...`) may suggest a sarcastic or literary style. Removing punctuation would discard these valuable features.

!!! info "Why lemmatize instead of stem?"
    Lemmatization (via spaCy) produces valid dictionary words (`better` → `good`), while stemming (e.g., Porter) produces truncated forms (`better` → `better`, `running` → `run`). Lemmatization is preferred for TF-IDF features because it preserves word meaning and produces cleaner n-grams.

!!! info "Why en_core_web_sm?"
    The `en_core_web_sm` model (~12 MB) provides tokenization, POS tagging, and lemmatization without loading word vectors. Since the TF-IDF pipeline doesn't need pre-trained word vectors, the small model is sufficient and faster. The embedding-based approach in Notebook 03 uses `en_core_web_md` (with 300-d vectors) instead.

---

## Consistency Between Training and Inference

The preprocessing function is defined in two places:

| File | Context |
|------|---------|
| `scripts/build_dataset.py` | Used during dataset creation to populate the `clean_text` column |
| `app/ml.py` | Used at inference time to preprocess user input before prediction |

Both implementations are functionally identical to ensure that the model receives input in the same format it was trained on.

!!! warning "Important"
    If the preprocessing logic is modified, both files must be updated simultaneously. Mismatched preprocessing between training and inference will degrade model performance.
