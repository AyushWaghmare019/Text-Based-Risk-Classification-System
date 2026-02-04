# Text-Based Spam Detection System (Machine Learning)

## Problem Statement
Spam messages cause financial loss, security risks, and poor user experience.  
The goal of this project is to build an end-to-end machine learning system that classifies SMS messages as Spam or Ham (Not Spam) using classical machine learning techniques, with a strong focus on imbalanced data handling and decision-making.

## Dataset
Source: SMS Spam Collection Dataset  
Size: ~5,500 SMS messages  

Class Distribution:
- Ham (0): ~87%
- Spam (1): ~13% (imbalanced)

## Key Challenges
- Highly imbalanced target variable
- Unstructured text data
- Avoiding data leakage
- Choosing correct evaluation metrics (accuracy is misleading)
- Making probability-based decisions instead of hard rules

## Exploratory Data Analysis (EDA)
EDA was performed in a separate notebook (eda.ipynb) to understand data characteristics and guide modeling decisions.

Key insights:
- Spam messages are generally longer than ham messages
- Spam contains more numbers, special characters, and URLs
- Vocabulary usage differs significantly between spam and ham
- Numeric features are right-skewed
- Spam messages show higher intra-class similarity, indicating templated behavior

## Feature Engineering

Text Features:
- Lowercased raw text
- TF-IDF vectorization
- Unigrams and bigrams (1,2)
- Stopword removal
- Frequency-based filtering

Numeric / Structural Features:
- message_length: total characters in SMS
- word_count: number of words
- number_count: count of numeric tokens
- contains_number: binary indicator for digits
- special_char_count: number of symbols
- has_url: binary URL indicator

Text and numeric features were combined using sparse matrix concatenation.

## Models Used

Logistic Regression (Baseline):
- Used as a strong baseline for sparse TF-IDF features
- Interpretable and fast
- Handles class imbalance using class_weight='balanced'

XGBoost (Final Model):
- Captures non-linear feature interactions
- Handles class imbalance using scale_pos_weight
- Used as the final optimized model

Models not used:
- Linear Regression: not suitable for classification
- KNN: inefficient for high-dimensional sparse text
- Naive Bayes: weaker calibration compared to Logistic Regression
- Deep Learning models: dataset too small, unnecessary complexity
- SMOTE / Upsampling: distorts text feature space

## Evaluation Strategy
Due to class imbalance, accuracy was not used as the primary metric.

Metrics used:
- Precision
- Recall
- F1-score
- Precision–Recall Curve

Spam class performance summary:
- Logistic Regression: Precision 0.88, Recall 0.94, F1-score 0.91
- XGBoost: Precision 0.91, Recall 0.92, F1-score 0.92

## Threshold Tuning
Instead of using the default 0.5 probability cutoff, the decision threshold was tuned using the Precision–Recall curve.

- Final threshold selected: 0.42
- Improved balance between spam recall and precision
- Demonstrates decision-aware modeling beyond default settings

## Final Model Selection
XGBoost was selected as the final model due to its better balance between precision and recall on the spam class.  
Logistic Regression remains a strong alternative when interpretability or lower latency is required.

## Project Structure
Text-Based-Risk-Classification-System/
├── data/
│   └── spam.csv
├── notebooks/
│   ├── eda.ipynb
│   └── modeling.ipynb
├── models/
│   ├── xgboost.pkl
│   ├── logistic_regression.pkl
│   ├── tfidf.pkl
│   └── scaler.pkl
├── README.md

## Key Takeaways
- Full end-to-end machine learning pipeline
- Proper handling of imbalanced data without blind resampling
- Feature engineering combining semantic and structural signals
- Evaluation based on precision, recall, and F1-score
- Decision threshold tuning using PR curve
- Clear separation of EDA and modeling

## Resume Highlight
Built an end-to-end SMS spam detection system using TF-IDF, engineered text and numeric features, Logistic Regression and XGBoost, with precision–recall optimization and decision threshold tuning.
