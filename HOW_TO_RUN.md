# HOW TO RUN

This folder contains everything needed — the dataset is already included.

```
spam-detection-notebook/
├── data/
│   └── spam.csv                     ← the only dataset used
├── spam_detection_pipeline.ipynb    ← the whole project: code + outputs, already run
└── requirements.txt
```

## Option A — Just view it (fastest)
Open `spam_detection_pipeline.ipynb` in Jupyter, VS Code, or GitHub — every
plot, table, and metric is already saved inside it from the last run. You
don't need to run anything to see the results.

## Option B — Run it yourself

```bash
pip install -r requirements.txt
jupyter notebook spam_detection_pipeline.ipynb
```

Then in Jupyter: **Kernel → Restart & Run All**. Takes a few minutes
(SHAP and Optuna tuning are the slowest steps). All 31 code cells will
re-execute and regenerate every plot/table/metric in place.

## What's inside the notebook

1. Setup
2. Load & Clean Data
3. Exploratory Data Analysis (class imbalance, feature distributions, correlation heatmap, outliers, top words, SVD projection)
4. Leakage-safe train/test split & feature engineering (TF-IDF + structural features)
5. Model comparison — 8 models (Logistic Regression, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, XGBoost, LightGBM, CatBoost)
6. Class imbalance strategy comparison (none / class_weight / SMOTE / Borderline-SMOTE / ADASYN / NearMiss)
7. Hyperparameter tuning (RandomizedSearchCV for Logistic Regression, Optuna for XGBoost)
8. Evaluation suite (ROC, PR curve, confusion matrix, classification report, decision threshold, learning curve, validation curve)
9. SHAP explainability (summary plot, importance bar, waterfall)
10. Saves trained models to `models/`
11. Predicts on new example messages
12. Summary of findings + fixed issues + future improvements
