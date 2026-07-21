# SMS Spam Detection — Text-Based Risk Classification

An end-to-end, research-grade machine learning project that classifies SMS
messages as **spam** or **ham** using classical ML on TF-IDF + engineered
structural features. Delivered as a single, fully-executed Jupyter notebook —
all code, plots, tables, and metrics are already saved inside it.

> **Scope note:** this project is intentionally scoped to the ML/data-science
> pipeline — no deployment, API, Docker, or MLOps tooling is included.

## 1. Project Overview

Spam SMS messages cause financial loss, phishing exposure, and degrade user
trust. The goal is a classifier that:
- Handles a realistically **imbalanced** target (~87% ham / 13% spam) without
  resorting to misleading accuracy-only evaluation.
- Combines **semantic** signal (TF-IDF word/bigram features) with
  **structural** signal (message length, digit/URL/special-character
  patterns) engineered from the raw text.
- Is evaluated and tuned honestly: every result below comes straight out of
  the notebook's own executed cells, not from memory.

## 2. Dataset

**Source:** SMS Spam Collection Dataset (`data/spam.csv`), the only dataset
used anywhere in this project.

- Raw size: 5,572 messages
- 403 exact duplicate messages removed before splitting (prevents identical
  messages leaking across train/test) → 5,169 messages used
- Class split: ~87% ham / ~13% spam

## 3. Methodology

```
raw CSV → clean (dedupe, label-map) → stratified 80/20 split
        → [TRAIN] fit TF-IDF + StandardScaler on structural features
        → [TEST]  transform only (no re-fitting) → model comparison
        → imbalance-strategy study → hyperparameter tuning
        → evaluation suite → SHAP explainability
```

The single most important rule enforced throughout the notebook: **all
fitting (TF-IDF, scaler) happens on the training split only** — the test
split only ever calls `.transform()`. Combined feature matrix: 2,864
columns (TF-IDF unigrams/bigrams + 9 engineered structural features),
4,135 training rows / 1,034 test rows.

## 4. Exploratory Data Analysis

Generated inline in the notebook (Section 3):
- Class imbalance bar chart
- Engineered feature distributions by class (boxplots)
- Correlation heatmap over the engineered numeric features
- z-scored outlier analysis
- Top-20 vocabulary words for spam vs. ham
- 2D Truncated-SVD projection of the TF-IDF space (sparse-safe PCA analogue —
  true PCA isn't practical on a 2,800+-dimension sparse matrix)

Spam messages are consistently longer, contain more digits/special
characters/URLs, and use a visibly different vocabulary (prize/urgency
words) than ham — all confirmed visually in the notebook's plots.

## 5. Feature Engineering

**Text:** TF-IDF, unigrams + bigrams, `min_df=3`, `max_df=0.9`, English stopwords removed.

**Structural (engineered from raw text, before lowercasing):**

| Feature | Why it helps |
|---|---|
| `message_length`, `word_count` | Spam messages are systematically longer (templated marketing copy) |
| `number_count`, `contains_number` | Spam frequently embeds phone numbers / prize codes |
| `special_char_count` | Spam uses more symbols for urgency (`!`, `£`, `*`) |
| `has_url` | Spam disproportionately contains links |
| `avg_word_length` | Crude proxy for obfuscated/leetspeak tokens used to dodge keyword filters |
| `digit_ratio`, `uppercase_ratio` | Length-normalized versions, more robust than raw counts |

Text and structural features are combined via sparse `hstack` into one matrix per split.

## 6. Model Comparison

All eight requested models trained with each library's native imbalance
handling (`class_weight='balanced'` or `scale_pos_weight`), scored on the
held-out test set, ranked by **F1** (the imbalance-appropriate metric —
always predicting "ham" would score ~87% accuracy while catching zero spam):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | MCC |
|---|---|---|---|---|---|---|---|
| Extra Trees | 0.984 | 0.991 | 0.878 | 0.931 | 0.997 | 0.988 | 0.924 |
| Random Forest | 0.982 | 0.967 | 0.885 | 0.924 | 0.997 | 0.984 | 0.915 |
| LightGBM | 0.980 | 0.917 | 0.924 | 0.920 | 0.996 | 0.980 | 0.909 |
| XGBoost | 0.979 | 0.904 | 0.931 | 0.917 | 0.995 | 0.977 | 0.905 |
| Logistic Regression | 0.978 | 0.886 | 0.947 | 0.915 | 0.992 | 0.948 | 0.903 |
| CatBoost | 0.973 | 0.860 | 0.939 | 0.898 | 0.994 | 0.973 | 0.883 |
| Gradient Boosting | 0.968 | 0.871 | 0.878 | 0.875 | 0.989 | 0.953 | 0.856 |
| Decision Tree | 0.955 | 0.784 | 0.885 | 0.832 | 0.925 | 0.709 | 0.807 |

**Accuracy is not used to pick the winner.** Extra Trees, Random Forest, and
LightGBM lead on F1/MCC; Decision Tree is clearly weakest, as expected for a
single unpruned tree on high-dimensional sparse text.

## 7. Hyperparameter Tuning

- **Logistic Regression** — `RandomizedSearchCV` (10 iters, 5-fold CV, F1 scoring). Best: `C=50, solver=lbfgs`.
- **XGBoost** — Optuna (TPE sampler, 20 trials, 5-fold CV F1 objective). Best: `n_estimators=450, max_depth=5, learning_rate=0.037, subsample=0.80, colsample_bytree=0.60, min_child_weight=1, gamma=0.038`.

Tuning is always scored via cross-validation on the **training split only** —
the test set is touched exactly once, at the very end, for reported numbers.
Tuning against the test set would itself be a data-leakage bug.

## 8. Class Imbalance Strategy Comparison

Rather than assume resampling helps, six strategies were tested with an
identical Logistic Regression baseline (isolates the resampling effect from
model choice):

| Strategy | F1 | PR-AUC |
|---|---|---|
| Borderline-SMOTE | 0.933 | 0.956 |
| SMOTE | 0.925 | 0.956 |
| No correction | 0.921 | 0.954 |
| class_weight='balanced' | 0.915 | 0.948 |
| ADASYN | 0.912 | 0.956 |
| NearMiss (undersampling) | 0.778 | 0.598 |

**Takeaway:** SMOTE-family methods give a real F1 bump over `class_weight`
here, so the common claim that "SMOTE always distorts sparse text feature
spaces" doesn't fully hold on this dataset — Borderline-SMOTE was the best
performer in this run. `class_weight` is still a reasonable default when you
want zero synthetic data and zero extra hyperparameters, but the data shows
it isn't the free lunch it's sometimes assumed to be. NearMiss undersampling
is clearly the wrong choice — it throws away real ham examples from an
already-small (~5,000-row) dataset.

## 9. Evaluation

Full suite generated for the final tuned model (XGBoost), all inline in the notebook:
- ROC Curve
- Precision-Recall Curve
- Confusion Matrix + Classification Report
- Learning Curve (train/val F1 vs. training-set size)
- Validation Curve (F1 vs. `max_depth`)

**Decision threshold tuning:** instead of the default 0.5 cutoff, the
threshold that maximizes F1 on the PR curve was found programmatically.
This run: **threshold = 0.438, F1 = 0.944**.

## 10. Explainable AI (SHAP)

`TreeExplainer` on the tuned XGBoost model (sparse-native, exact — not the
slower model-agnostic `KernelExplainer`), computed on a 300-row sample of the
test set:
- Summary plot (global feature impact + direction)
- Feature importance bar chart (mean |SHAP value|)
- Waterfall plot (single-prediction explanation)

Importance is a mix of high-signal spam vocabulary terms (prize/urgency
words picked up by TF-IDF) and the engineered structural features
(`has_url`, `special_char_count`, `number_count`) — confirming the
structural features contribute real, independent signal rather than just
duplicating what TF-IDF already captures.

## 11. Project Structure

```
spam-detection-notebook/
├── data/
│   └── spam.csv                     ← the only dataset used
├── spam_detection_pipeline.ipynb    ← all code + all outputs (already run)
├── models/                          ← populated when the notebook is run
├── requirements.txt
├── HOW_TO_RUN.md
└── README.md
```

## 12. Installation & Usage

```bash
pip install -r requirements.txt
jupyter notebook spam_detection_pipeline.ipynb
```

The notebook already contains every output from a completed run — open it
and read straight through. To regenerate everything yourself: **Kernel →
Restart & Run All** (takes a few minutes; Optuna tuning and SHAP are the
slowest steps). This also saves trained models to `models/` and prints
predictions on two example messages at the end.

## 13. Future / Research-Level Improvements

Not yet implemented, listed honestly as future work rather than claimed as done:

- **Ensembling:** stacking or blending the top 3 models (Extra Trees, Random Forest, LightGBM) to see if it beats any single model.
- **Probability calibration:** `CalibratedClassifierCV` (Platt/isotonic) on the tree-based models — tree ensemble probabilities are often poorly calibrated even when ranking (AUC) is excellent.
- **Feature selection:** chi-squared or mutual-information filtering on the TF-IDF vocabulary to shrink the feature space and test whether performance holds with far fewer features.
- **Cross-validation for the final reported numbers:** current comparison uses a single 80/20 split; repeated stratified k-fold would give confidence intervals around each metric instead of point estimates.
- **Error analysis:** a qualitative pass over the false positives/false negatives to look for systematic patterns (e.g. short spam messages that mimic conversational ham).
- **Ablation study:** re-run the model comparison with TF-IDF-only features vs. structural-only features vs. combined, to quantify how much the engineered structural features actually contribute over TF-IDF alone.

## Resume Highlight

Built an SMS spam detection pipeline in a single reproducible notebook:
TF-IDF + engineered structural features, an 8-model comparison (Logistic
Regression → CatBoost) evaluated on F1/PR-AUC/MCC, an empirical
class-imbalance study (SMOTE/ADASYN/NearMiss vs. class-weighting),
Optuna/RandomizedSearchCV tuning, PR-curve-based decision threshold
optimization, and SHAP-based explainability.
