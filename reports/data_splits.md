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
| **Train** | 632 | 63.3% | `6ce3bb6779f10d46616205a01332425710c69a39e4dbb6ee51f656a2d81e4abd` |
| **Validation** | 158 | 15.8% | `adf6f515f61bd76c049104f45d448e680ec6607eab52cd9c0b9258055a2dfce1` |
| **Test** | 208 | 20.8% | `f02a659569eaf58ca8bfa628f9ab1bbc4293d2a6c40556a6ce6340ddba976439` |

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
