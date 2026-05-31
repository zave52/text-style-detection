# Results Comparison

This page consolidates the results of both modeling approaches for easy comparison.

---

## Summary Table

| Approach | Model | Target | Test Accuracy | Test F1 | CV F1 |
|----------|-------|--------|---------------|---------|-------|
| TF-IDF + LinearSVC | LinearSVC | Style | **0.890** | **0.886** (weighted) | **0.918** (weighted) |
| TF-IDF + LinearSVC | LinearSVC | Tone | **0.940** | **0.935** (weighted) | **0.950** (weighted) |
| Embeddings + MultiOutput | SVM (RBF) | Style | 0.870 | — | 0.853 (macro) |
| Embeddings + MultiOutput | SVM (RBF) | Tone | 0.890 | — | 0.872 (macro) |
| Embeddings + MultiOutput | Random Forest | Style | 0.855 | — | 0.837 (macro) |
| Embeddings + MultiOutput | Random Forest | Tone | 0.880 | — | 0.846 (macro) |

!!! note "Metric Difference"
    Approach 1 reports **F1-weighted**, while Approach 2 reports **F1-macro**. Since the dataset is perfectly balanced (200 samples per class), these values are close but not directly comparable. F1-macro gives equal weight to all classes regardless of support, while F1-weighted weights by support.

---

## Style Classification Comparison

### TF-IDF + LinearSVC

| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| academic | 0.97 | 0.93 | 0.95 |
| business | 0.71 | 0.85 | 0.77 |
| formal | 0.88 | 0.72 | 0.79 |
| informal | 0.90 | 0.95 | 0.93 |
| literary | 1.00 | 0.97 | 0.99 |

### Embeddings + SVM (RBF)

| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| academic | 0.95 | 0.88 | 0.91 |
| business | 0.78 | 0.80 | 0.79 |
| formal | 0.82 | 0.82 | 0.82 |
| informal | 0.83 | 0.95 | 0.88 |
| literary | 1.00 | 0.90 | 0.95 |

### Key Observations

- **TF-IDF wins on most classes** for style classification, particularly `academic` (0.95 vs 0.91) and `literary` (0.99 vs 0.95).
- **Embeddings slightly improve `formal`** (0.82 vs 0.79) and `business` (0.79 vs 0.77), where semantic similarity matters more.
- The `business` vs `formal` confusion is present in both approaches, suggesting these styles share structural overlap.

---

## Tone Classification Comparison

### TF-IDF + LinearSVC

| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| aggressive | 1.00 | 0.90 | 0.95 |
| friendly | 0.93 | 0.93 | 0.93 |
| neutral | 0.91 | 0.97 | 0.94 |
| sarcastic | 0.88 | 0.93 | 0.90 |
| urgent | 0.97 | 0.95 | 0.96 |

### Embeddings + SVM (RBF)

| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| aggressive | 0.95 | 0.88 | 0.91 |
| friendly | 0.88 | 0.88 | 0.88 |
| neutral | 0.85 | 0.90 | 0.87 |
| sarcastic | 0.85 | 0.88 | 0.86 |
| urgent | 0.93 | 0.93 | 0.93 |

### Key Observations

- **TF-IDF outperforms embeddings on all tone classes**, with an especially large gap on `sarcastic` (0.90 vs 0.86) and `aggressive` (0.95 vs 0.91).
- Tone detection relies heavily on **lexical cues** (specific words like "immediately", "please", "obviously") that TF-IDF captures well through n-grams.
- Averaging word vectors loses word-order information, which can be important for tone.

---

## Why TF-IDF Outperforms Embeddings

Several factors explain why the simpler TF-IDF approach outperforms the embedding-based approach on this dataset:

1. **Dataset size**: With only 1 000 samples, there isn't enough data to fully exploit the semantic richness of pre-trained embeddings. TF-IDF learns task-specific features directly.

2. **Information loss from averaging**: The `doc.vector` (mean of token vectors) discards word order and relative position. TF-IDF with n-grams `(1,3)` preserves some sequential information.

3. **Lexical specificity**: Style and tone are often signaled by **specific words and phrases** rather than semantic similarity. TF-IDF directly captures these lexical indicators, while embeddings map similar words to similar vectors (potentially losing discriminative power).

4. **Linear separability**: With high-dimensional TF-IDF features, classes may be more linearly separable, allowing the simple `LinearSVC` to work effectively.

---

## Chosen Model for Deployment

!!! success "Deployed Model"
    The **TF-IDF + LinearSVC** approach is used for the production inference API, as it achieves higher performance with lower computational cost and no dependency on large embedding models at inference time.

| Property | Value |
|----------|-------|
| Approach | TF-IDF + LinearSVC |
| Style F1 (weighted) | 0.886 |
| Tone F1 (weighted) | 0.935 |
| Model files | `saving/style_model.joblib`, `saving/tone_model.joblib` |
| Inference dependency | spaCy `en_core_web_sm` (for preprocessing only) |
