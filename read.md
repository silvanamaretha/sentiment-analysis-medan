# Sentiment Analysis of Public Opinion on Medan City Government

Old project — Silvana Maretha Br. Simbolon, Universitas Negeri Medan (2023)

---

## Overview

This project analyzes public sentiment toward the Medan City Government using Twitter data. The goal is to classify tweets into three sentiment categories: Positive, Neutral, and Negative.

The initial version of this project used TextBlob (via English translation) for labeling, which introduced significant noise due to translation artifacts and the lexicon-based nature of TextBlob. This improved version replaces TextBlob with IndoBERT, a RoBERTa-based model pre-trained on Indonesian text, resulting in more contextually accurate labels and substantially better model performance.

---

## Pipeline

```
Scrapping Data → Preprocessing → nlp.csv → Modeling (NB & SVM) → Evaluation
```

1. Preprocessing: Case folding, cleaning (mentions, hashtags, URLs, symbols), slang normalization, stopword removal (NLTK), stemming (Sastrawi)
2. Labeling: Automatic sentiment labeling using IndoBERT (`w11wo/indonesian-roberta-base-sentiment-classifier`)
3. Feature Extraction:TF-IDF with bigrams (ngram_range=(1,2), max_features=5000)
4. Oversampling: SMOTE applied only on training data to handle class imbalance
5. Modeling: Multinomial Naive Bayes (alpha=0.1) and LinearSVC
6. Evaluation: Precision, Recall, F1-Score on unseen test data

---

## Dataset

- Source: Twitter (scraped using Tweet-Harvest)
- Query: Tweets mentioning pemerintah kota Medan
- Total data: 596 tweets (after deduplication)
- Label distribution (IndoBERT):

| Sentiment | Count |
| --------- | ----- |
| Neutral   | 353   |
| Negative  | 170   |
| Positive  | 73    |

---

## Results

### Naive Bayes (Multinomial, alpha=0.1)

| Class        | Precision | Recall | F1-Score |
| ------------ | --------- | ------ | -------- |
| Negative     | 0.71      | 0.85   | 0.77     |
| Neutral      | 0.87      | 0.82   | 0.84     |
| Positive     | 0.50      | 0.40   | 0.44     |
| **Accuracy** |           |        | **0.78** |

### SVM (LinearSVC)

| Class        | Precision | Recall | F1-Score |
| ------------ | --------- | ------ | -------- |
| Negative     | 0.77      | 0.68   | 0.72     |
| Neutral      | 0.78      | 0.90   | 0.84     |
| Positive     | 0.62      | 0.33   | 0.43     |
| **Accuracy** |           |        | **0.77** |

Naive Bayes slightly outperforms SVM on this dataset, which is consistent with the known behavior of Naive Bayes on small, sparse text data.

---

## Key Findings

- Class imbalance was a significant challenge, SMOTE was applied exclusively on training data to avoid data leakage
- Positive sentiment remains the weakest class (F1=0.44) due to limited data variety even after oversampling
- IndoBERT labeling significantly improved performance compared to the original TextBlob approach (test accuracy: 54% → 78%)

---

## What the Data Says

Based on IndoBERT labeling of 596 tweets mentioning the Medan City Government:

- **59% Neutral** — mostly informational tweets, event announcements, and factual reporting
- **29% Negative** — dominated by criticism around vigilantism (premanisme), illegal parking, drainage issues, and lack of government transparency
- **12% Positive** — appreciation for government programs, mostly formal/institutional accounts

Key insight: Public sentiment on Twitter leans neutral-to-negative. The negative discourse is concentrated around public safety and organized crime tolerance, not basic services. Positive sentiment is notably scarce and largely driven by official accounts rather than organic public opinion.

---

## Limitations

- Small dataset: 596 tweets limits model generalization
- Automatic labeling: IndoBERT labels are more accurate than TextBlob but still not as reliable as manual annotation
- Single domain: Data only covers Medan City Government tweets; generalization to other domains is limited
- Positive class underperformance: Low F1 for Positive class reflects insufficient data variety, not solvable by SMOTE alone

---

## How to Run

All notebooks were developed using Google Colab.

1. Upload `full_text.csv` or your raw data and run `Preprocessing_Data.ipynb` to generate `nlp.csv`
2. Upload `nlp.csv` and run `Naive Bayes and SVM model.ipynb` for modeling and evaluation

Note: IndoBERT labeling in `Naive Bayes and SVM model.ipynb` requires internet connection and may take a few minutes for 596 samples.

---

## Tech Stack

Python · Pandas · NumPy · NLTK · Sastrawi · Scikit-learn · Imbalanced-learn · HuggingFace Transformers · Matplotlib · Seaborn
