# Data Splits and Leakage Prevention Report

## Splitting Strategy

- **Method:** Nested 5-fold `StratifiedGroupKFold` on combined `style__tone` strata.
- **Near-duplicate grouping:** Word trigram Jaccard similarity ≥ 0.82 or containment ≥ 0.95.
- **Unique groups:** 968 groups across 998 clean texts.
- **Random seed:** `42` (outer fold seed=42, inner fold seed=43).
- **Disjointness:** Zero overlap between splits for `sample_id`, `group_id`, and normalized text.

## Split Summary

| Split | Rows | Percentage | SHA-256 Hash |
|---|---:|---:|---|
| **Train** | 632 | 63.3% | `bf70d74eb7775b387867c7b94c5cffe74b8488fd711229c1c880f3ce6b29af23` |
| **Validation** | 158 | 15.8% | `a1886267233bdf51109731ce90a3dfc6bf2ac94d2f1fd8c5e10aac64f654a7da` |
| **Test** | 208 | 20.8% | `bf82c29cbecb9ea2479c276f6152ad8ff8f8ba942da299e93d31a7d4c098d796` |

## Class Distributions Across Splits

### Writing Styles

| Style | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| `academic` | 128 | 32 | 40 | 200 |
| `business` | 120 | 30 | 49 | 199 |
| `formal` | 128 | 32 | 39 | 199 |
| `informal` | 128 | 32 | 40 | 200 |
| `literary` | 128 | 32 | 40 | 200 |

### Communicative Tones

| Tone | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| `aggressive` | 125 | 35 | 40 | 200 |
| `friendly` | 127 | 33 | 40 | 200 |
| `neutral` | 129 | 31 | 40 | 200 |
| `sarcastic` | 129 | 31 | 40 | 200 |
| `urgent` | 122 | 28 | 48 | 198 |
