📩 Text-Based Spam Detection System (Machine Learning)
🔍 Problem Statement

Spam messages cause financial loss, security risks, and poor user experience.
The goal of this project is to build an end-to-end machine learning system that classifies SMS messages as Spam or Ham (Not Spam) using classical ML techniques with a strong focus on imbalanced data handling and decision-making.

📊 Dataset

Source: SMS Spam Collection Dataset

Size: ~5,500 SMS messages

Classes:

Ham (0): ~87%

Spam (1): ~13% (imbalanced)

🧠 Key Challenges

Highly imbalanced target variable

Unstructured text data

Avoiding data leakage

Choosing correct evaluation metrics (accuracy is misleading)

Making probability-based decisions, not hard rules

🔎 Exploratory Data Analysis (EDA)

EDA was performed in a separate notebook (eda.ipynb) to understand data characteristics and guide modeling decisions.

Key Insights:

Spam messages are generally longer than ham messages

Spam contains more numbers, special characters, and URLs

Vocabulary usage differs significantly between spam and ham

Numeric features are right-skewed

Spam messages show higher intra-class similarity, indicating templated behavior

These insights directly informed feature engineering and model selection.

🛠 Feature Engineering
🔹 Text Features

Lowercased raw text

TF-IDF Vectorization

Unigrams + Bigrams (1,2)

Stopword removal

Frequency-based filtering

🔹 Numeric / Structural Features
Feature	Description
message_length	Total characters in SMS
word_count	Number of words
number_count	Count of numeric tokens
contains_number	Binary flag for digits
special_char_count	Count of symbols
has_url	Binary URL indicator

Text and numeric features were combined using sparse matrix concatenation.

🤖 Models Used
✅ Logistic Regression (Baseline)

Strong performance on sparse TF-IDF features

Interpretable

Used to validate feature quality

Handled imbalance using class_weight='balanced'

✅ XGBoost (Final Model)

Captures non-linear feature interactions

Handles class imbalance via scale_pos_weight

Used as the final optimized model

❌ Models Not Used (and Why)

Linear Regression: Not suitable for classification

KNN: Inefficient for high-dimensional sparse text

Naive Bayes: Weaker calibration than Logistic Regression

Deep Learning (BERT/LSTM): Dataset too small, unnecessary complexity

SMOTE / Upsampling: Distorts text feature space

📈 Evaluation Strategy

Due to class imbalance, accuracy was not used as the primary metric.

Metrics Used:

Precision

Recall

F1-score

Precision–Recall Curve

Results Summary (Spam Class):
Model	Precision	Recall	F1-score
Logistic Regression	0.88	0.94	0.91
XGBoost	0.91	0.92	0.92
🎯 Threshold Tuning (Decision-Aware ML)

Instead of using the default 0.5 probability cutoff, the decision threshold was tuned using the Precision–Recall curve.

Final threshold chosen: 0.42

Improved recall–precision balance for spam detection

Demonstrates real-world decision-making beyond model training

🏆 Final Model Selection

XGBoost was selected as the final model due to:

Better balance between precision and recall

Strong performance on mixed text + numeric features

Logistic Regression remains a strong alternative where interpretability or latency is critical.

📁 Project Structure
Text-Based-Risk-Classification-System/
│
├── data/
│   └── spam.csv
│
├── notebooks/
│   ├── eda.ipynb
│   └── modeling.ipynb
│
├── models/
│   ├── xgboost.pkl
│   ├── logistic_regression.pkl
│   ├── tfidf.pkl
│   └── scaler.pkl
│
├── README.md

🚀 Key Takeaways

Demonstrates full ML lifecycle

Handles imbalanced data correctly (no blind resampling)

Uses appropriate evaluation metrics

Applies decision threshold tuning

Clean separation of EDA and modeling

Production-aware feature pipeline
