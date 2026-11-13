# Comparing Classical Models and Neural Networks

This page compares TF-IDF + LinearSVC and spaCy embeddings + SVM/Random Forest with TextCNN, BiLSTM with attention, and DistilBERT. All approaches address the same task: predicting one of five writing styles and one of five tones for an English text.

**The figures below are historical results from different experiments. They are not the results of a shared, controlled test of all models.** The classical results come from the saved outputs of the original notebooks [1–2]; the neural results come from the published `artifacts/experiment.json` of the earlier deep learning implementation [3]. They do not describe the new local `artifacts/experiments/distilbert_01` run in this repository.

## 1. Preprocessing and Text Representation

| Approach | Preprocessing | Representation | Context and Word Order |
|---|---|---|---|
| TF-IDF + LinearSVC | `clean_text`: URL removal, spaCy `en_core_web_sm`, lemmatization, lowercasing | Sparse TF-IDF vector with up to 5000 features; vocabulary and IDF are fitted on train | Word n-grams capture local sequences, without a representation of full sentence context |
| spaCy embeddings + SVM/RF | Original `text`; spaCy `en_core_web_md`; vector averaging after removing stopwords and punctuation | Fixed, dense document vector with 300 components | Static vectors encode lexical similarity; averaging discards word order and positions |
| TextCNN | Vocabulary fitted on train only, `casefold`, an `<upper>` marker for uppercase words; punctuation is retained | Trainable embeddings and convolutions over windows of 3, 4, and 5 tokens | Learned local patterns; max pooling retains the strongest features but discards their absolute positions |
| BiLSTM + attention | The same type of learned vocabulary, truncation and padding; packed sequences | Trainable embeddings, bidirectional hidden states, attention pooling, and final hidden states | Each hidden state depends on the left and right context within the available sequence |
| DistilBERT | Its own cased subword tokenizer, special tokens, attention masks, and a configured sequence-length limit | A pretrained transformer encoder fine-tuned for style and tone classification | Self-attention combines distant tokens within the input limit; a word's representation depends on its surroundings |

### Important Implementation Details

In the classical TF-IDF approach, spaCy performs preprocessing rather than style or tone classification. `preprocess_text` preserves punctuation in the cleaned string, but the default word analyzer in `TfidfVectorizer` generally discards punctuation and single-character tokens. Preserving punctuation in `clean_text` therefore does not, by itself, mean that the classifier uses it as a feature. The best reported parameters are `C=1`, `max_features=5000`, and n-gram ranges of `(1,3)` for style and `(1,2)` for tone [1].

The spaCy notebook uses raw `text`. Its actual `SpacyEmbeddingVectoriser` removes tokens marked as `is_stop` or `is_punct` and averages the remaining vectors. If no suitable tokens remain, it falls back to `doc.vector` or a zero vector. The saved output confirms the use of `en_core_web_md` with 300-dimensional vectors; falling back to another spaCy model should not be treated as the same experiment. These vectors are not fine-tuned using the style and tone losses. Averaging and stopword removal can discard signals useful for negation, style, or tone. This is a property of the representation, not a demonstrated cause of any particular error [2].

CNN and BiLSTM learn embeddings from scratch on a small training set. DistilBERT starts with English language representations learned during pretraining and updates its encoder during fine-tuning. The comparison therefore reflects both architecture and different forms of pretraining. The spaCy vectors are also pretrained, but they are static and remain fixed in this experiment.

Neural context is limited by tokenization and truncation. In the published DL experiment reported here, CNN/BiLSTM used a limit of 256 tokens, while DistilBERT used 192 tokens, including special tokens [4]. The new shared pipeline records a separate `max_length` for each run; the relevant run configuration should be used to determine its limit.

## 2. Prediction Architecture

**TF-IDF + LinearSVC:** two independent sklearn pipelines. Each contains its own TF-IDF vectorizer and LinearSVC for one target. Training the style classifier does not change the representation or parameters of the tone classifier.

**spaCy + MultiOutputClassifier:** the same document vector is passed to two separate estimators. The notebook compares an RBF SVM (`C=10`, `gamma="scale"`) with a Random Forest containing 300 trees. `MultiOutputClassifier` is a wrapper that trains a classifier for each output; it does not create a shared trainable encoder or model dependencies between the labels [2, 7].

**Neural networks:** a shared encoder creates a document representation, and two classification heads each return five logits. Both tasks update the encoder through the sum of their cross-entropy losses. CNN uses masked pooling; BiLSTM uses packed sequences and attention; DistilBERT uses the representation of the first special token. Padded positions are excluded from the relevant computations. At inference time, separate validation-fitted temperatures are applied to the style and tone logits before softmax. A positive temperature changes probabilities while preserving the ordering of logits and the argmax class.

The architectural distinction is between jointly learning a representation and supplying the same fixed features to two independent classifiers.

## 3. Reported Classical Model Results

Accuracy is shown as a percentage, and F1 uses the `0–1` scale. The LinearSVC notebook prints test accuracy to only two decimal places on the `0–1` scale.

| Model | Task | Test Accuracy | Test F1 in the Saved Output | CV F1 |
|---|---|---:|---|---|
| TF-IDF + LinearSVC | Style | ≈89.00% | 0.8860, weighted | 0.9183, weighted |
| TF-IDF + LinearSVC | Tone | ≈94.00% | 0.9353, weighted | 0.9496, weighted |
| spaCy + RBF SVM | Style | 87.00% | ≈0.87, macro, rounded to 2 decimal places | 0.8534 ± 0.0215, macro |
| spaCy + RBF SVM | Tone | 89.00% | ≈0.89, macro, rounded to 2 decimal places | 0.8721 ± 0.0156, macro |
| spaCy + Random Forest | Style | 85.50% | ≈0.85, macro, rounded to 2 decimal places | 0.8365 ± 0.0187, macro |
| spaCy + Random Forest | Tone | 88.00% | ≈0.88, macro, rounded to 2 decimal places | 0.8461 ± 0.0131, macro |

Sources: `modelling_02.ipynb`, code cells 2–4, and `embedding_models_03.ipynb`, code cells 11, 15, 17, 19, and 25 [1–2]. Cell numbers refer to zero-based indices in the notebook JSON.

For spaCy, the more precise values in the CV column are not test F1 scores. They come from a separate five-fold procedure over all 1000 records. Test F1 in the saved holdout report is available only in rounded form; it cannot be recovered exactly from the CV mean or rounded per-class values.

## 4. Published Neural Experiment

The source is the machine-readable `artifacts/experiment.json` in the DL repository, rather than illustrative figures or a new training run [3]. All three neural models below were evaluated on the same 208 prepared test texts.

| Model | Task | Test Accuracy | Test Macro F1 | Test Weighted F1 |
|---|---|---:|---:|---:|
| TextCNN | Style | 84.13% | 0.8402 | 0.8403 |
| TextCNN | Tone | 89.42% | 0.8927 | 0.8942 |
| BiLSTM + attention | Style | 78.37% | 0.7816 | 0.7820 |
| BiLSTM + attention | Tone | 84.62% | 0.8420 | 0.8453 |
| DistilBERT | Style | **93.27%** | **0.9338** | **0.9327** |
| DistilBERT | Tone | **97.12%** | **0.9709** | **0.9712** |

| Model | Validation Mean Macro F1 | Test Joint Accuracy |
|---|---:|---:|
| TextCNN | 0.8835 | 78.37% |
| BiLSTM + attention | 0.8050 | 68.27% |
| DistilBERT | **0.9518** | **90.38%** |

DistilBERT correctly predicted style for 194 of the 208 texts, tone for 202, and both labels together for 188. Joint accuracy is not the mean of the two task accuracies. It cannot be recovered from the two individual accuracies without knowing which predictions were correct for both tasks.

DistilBERT was selected using validation mean macro F1, with the test set reserved for final evaluation. The value `val_mean_macro_f1=0.9548` from another local five-epoch run is a separate validation metric. It is neither test accuracy nor test F1 and does not replace the values from this published experiment.

## 5. Differences Between LinearSVC and DistilBERT

For scores on the `0–1` scale, the difference in percentage points is calculated as `100 × (score_DistilBERT − score_LinearSVC)`.

| Metric | LinearSVC | DistilBERT | Approximate Difference |
|---|---:|---:|---:|
| Style accuracy | ≈89.00% | 93.27% | **+4.27 pp** |
| Tone accuracy | ≈94.00% | 97.12% | **+3.12 pp** |
| Style weighted F1 | 0.8860 | 0.9327 | **+4.67 pp** on the percentage scale for F1 |
| Tone weighted F1 | 0.9353 | 0.9712 | **+3.59 pp** on the percentage scale for F1 |

The F1 comparison uses the more precise notebook values `0.8860 / 0.9353`, rather than the shortened `0.886 / 0.935` in the README. Accuracy differences are approximate because the classical accuracy figures were rounded in the source. Percentage points express an absolute difference, not a relative percentage improvement.

**These differences describe reported values from different runs. They do not measure a DistilBERT improvement on an identical controlled test set or establish statistical significance.** Weighted F1 is compared with weighted F1; DistilBERT macro F1 is not substituted for weighted F1 in the calculation.

The lower reported CNN/BiLSTM scores also show that replacing a classical classifier with a neural network does not automatically improve performance. In this particular DL experiment, pretrained DistilBERT achieved the best results.

## 6. Why Test Splits and Metric Averaging Matter

### Different Test Splits

- **LinearSVC:** the original 1000 records, with two separate calls to `train_test_split(test_size=0.2, random_state=42)`. Style is stratified by `y_style`, while tone is stratified by `y_tone`. Both holdouts contain 200 texts, but this does not guarantee the same rows for the two tasks. Both target-specific test reports have 40 examples per class.
- **spaCy:** the original 1000 records, with one shared 800/200 split stratified by style only. The test set has 40 examples per style, but tone supports are 36/44/40/42/38. CV runs separately over the full corpus for each target, rather than over those same 200 test texts.
- **DL:** correcting `literaly → literary` and excluding two conflicting records leaves 998 texts. Near-duplicate grouping and nested `StratifiedGroupKFold` using `style__tone` produce a shared split of **632 train / 158 validation / 208 test**. All neural networks use the same test IDs. Code verifies that IDs, groups, and normalized texts do not overlap across splits [5].

The same seed, the same original corpus, or the same test-set size does not establish that test texts are identical. Heuristic grouping reduces the risk of near-duplicate leakage but does not guarantee that all semantic similarity has been eliminated.

### Macro, Weighted, Mean Macro, and CV

For five classes:

$$
F1_{macro}=\frac{1}{5}\sum_{k=1}^{5}F1_k,
\qquad
F1_{weighted}=\sum_{k=1}^{5}\frac{n_k}{N}F1_k.
$$

Macro gives every class equal weight; weighted F1 accounts for each class's support. **When support is equal within a particular test set, macro and weighted F1 are mathematically equal.** This applies to the two balanced LinearSVC holdouts. Balance in the full corpus alone does not guarantee equality for an arbitrary split or target [8].

The DL test set is not balanced: style supports are 40/49/39/40/40, and tone supports are 40/40/40/40/48 in canonical label order. This is why, for example, DistilBERT style macro F1 of `0.9338` differs from weighted F1 of `0.9327`.

`mean_macro_f1` is the mean of the two target-level macro F1 scores: `(style_macro_f1 + tone_macro_f1) / 2`. It is not accuracy, joint accuracy, or weighted F1. Reports must also identify whether a metric was measured on validation or test.

CV F1 is a mean of fold scores, whereas test F1 describes a particular held-out set. Mean fold F1 does not necessarily equal F1 calculated from pooled predictions. The `±` values in the spaCy experiment are standard deviations across five folds; they are not confidence intervals for a difference between two models.

## 7. Additional Experiment Required for a Fair Comparison

The classical baselines must be **retrained on the same prepared train/validation/test splits used by the neural models**.

1. **Freeze the data.** Use one `data/prepared/manifest.json`, the same CSV files, `sample_id` values, and test-row order. For this corpus, use 632/158/208 without creating another random split. Apply the same label normalization and conflict exclusions to every model.
2. **Train TF-IDF + LinearSVC from scratch.** Reproduce the classical spaCy preprocessing for texts in the prepared splits. Fit the vocabulary and IDF only on the 632 training texts. If CV is used, keep the vectorizer inside the pipeline and use group-aware folds within train only. Select the configuration using the shared validation criterion, keeping test outside model selection.
3. **Retrain spaCy + SVM/RF.** Pin the `en_core_web_md` version, the 300-dimensional representation, the stopword/punctuation policy, and classifier settings. Fit classifiers only on prepared train and choose parameters using the same validation metric.
4. **Apply a shared selection protocol.** Use validation mean macro F1 as the criterion for all approaches. Fit CNN/BiLSTM vocabularies on train only and fine-tune DistilBERT on train. Use validation for selection, early stopping, and calibration; keep test outside those decisions. Record tokenizers, sequence-length limits, and truncation rates. Using the same source documents does not require replacing the pretrained tokenizer with one common vocabulary for every architecture.
5. **Evaluate all frozen models on the same 208 test rows.** For each task, calculate accuracy, macro F1, and weighted F1 using the same order of five classes and `zero_division=0`. Calculate joint accuracy from both labels of each identical row. Export per-class reports, confusion matrices, predictions, and business/formal errors.
6. **Optionally compare probabilities and stability.** LinearSVC does not provide `predict_proba` directly: NLL/Brier/ECE require calibration using train/validation without access to test. Use the already fixed validation temperatures for the neural models. Save configurations, model versions, weight checksums, and the dataset manifest. Multiple training seeds on a fixed holdout help assess variability; significance claims require a separate paired uncertainty assessment.

**Simply loading the old `joblib` files and evaluating them on the new prepared test set would be insufficient.** Some of those texts may have been included in the old 800-row LinearSVC training sets. Moreover, cell 27 of `embedding_models_03.ipynb` explicitly runs `best_pipeline.fit(X, y)` on **all 1000 records** before saving the pipeline. Those saved weights have already seen the prepared test texts and cannot serve as a fair baseline for this experiment [2].

The existing `scripts.evaluate` command verifies and compares neural bundles. A complete controlled comparison also needs adapters for newly trained sklearn pipelines so that their predictions are passed to the same metric calculations. The shared ML/DL experiment described here has not yet been performed; the historical table is not a substitute for it.

## Sources and Reproducibility

The classical implementation was checked at commit `0acde78c0242ce6c0898600433a95d71ede231e9`, and the published DL experiment at `6a63ac9d2c7a82a2c71cb43544b09924778dad03`. The links below are pinned to those snapshots.

1. [TF-IDF notebook: split code, GridSearchCV, and saved results](https://github.com/zave52/text-style-detection/blob/0acde78c0242ce6c0898600433a95d71ede231e9/modelling_02.ipynb).
2. [spaCy notebook: vectorizer, holdout, CV, and final training on all data](https://github.com/zave52/text-style-detection/blob/0acde78c0242ce6c0898600433a95d71ede231e9/embedding_models_03.ipynb).
3. [DL experiment.json: exact test metrics, confusion matrices, and validation criterion](https://github.com/PavloMolytovnyk/text_style_detection_deep_learning/blob/6a63ac9d2c7a82a2c71cb43544b09924778dad03/artifacts/experiment.json).
4. [DL README: architectures and configuration of the published experiment](https://github.com/PavloMolytovnyk/text_style_detection_deep_learning/blob/6a63ac9d2c7a82a2c71cb43544b09924778dad03/README.md).
5. [DL dataset manifest: 998 texts, 968 groups, 632/158/208 split, and class supports](https://github.com/PavloMolytovnyk/text_style_detection_deep_learning/blob/6a63ac9d2c7a82a2c71cb43544b09924778dad03/data/prepared/manifest.json).
6. [Current tokenization implementation](https://github.com/zave52/text-style-detection/blob/0acde78c0242ce6c0898600433a95d71ede231e9/app/tokenization.py), [legacy preprocessing](https://github.com/zave52/text-style-detection/blob/0acde78c0242ce6c0898600433a95d71ede231e9/app/ml.py).
7. [scikit-learn: MultiOutputClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html).
8. [scikit-learn: F1 and averaging methods](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html).
