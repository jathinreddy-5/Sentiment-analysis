Here is the exact content to paste:

```
# 🐦 Twitter Sentiment Analysis

> Binary sentiment classification of tweets using NLP and Logistic Regression on the Sentiment140 dataset.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Dataset](https://img.shields.io/badge/Dataset-Sentiment140-green)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-77.7%25-brightgreen)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-yellow)

---

## 📌 Overview

This project classifies tweets as **positive** or **negative** using the full ML pipeline:
- Data loading → Text cleaning → Feature extraction → Model training → Inference

| Metric | Value |
|---|---|
| Total Tweets | 1,600,000 |
| Training Accuracy | 79.9% |
| Test Accuracy | 77.7% |
| TF-IDF Features | 461,488 |

---

## 📁 Project Structure

```
📦 twitter-sentiment
 ┣ 📓 Twitter_sentiment.ipynb   # Main notebook
 ┣ 🔑 kaggle.json               # Kaggle API key (not committed)
 ┣ 🗜️ sentiment140.zip          # Raw dataset (auto-downloaded)
 ┗ 💾 trained_model.sav         # Saved Logistic Regression model
```

---

## ⚙️ Pipeline

```
1. Download Dataset      →   Kaggle API (sentiment140)
2. Load & Inspect        →   Pandas DataFrame (1.6M rows, 6 columns)
3. Label Encoding        →   {0, 4}  →  {0, 1}  (balanced 800K each)
4. Text Preprocessing    →   Remove noise → lowercase → stopwords → Porter stemming
5. Train/Test Split      →   80% train / 20% test (stratified)
6. TF-IDF Vectorization  →   461,488 sparse features
7. Train Model           →   Logistic Regression (max_iter=1000)
8. Save Model            →   pickle → trained_model.sav
```

---

## 🚀 Quick Start

**1. Clone the repo and open the notebook in Google Colab**

**2. Upload your `kaggle.json` API key to the Colab session**

**3. Run all cells — the dataset downloads automatically**

```python
!pip install kaggle
!kaggle datasets download -d kazanova/sentiment140

import nltk
nltk.download('stopwords')
```

---

## 🔍 Inference

```python
import pickle

# Load saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# Predict on a sample
prediction = loaded_model.predict(x_new)

if prediction[0] == 0:
    print('Negative tweet 😞')
else:
    print('Positive tweet 😊')
```

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading and wrangling |
| `numpy` | Numerical operations |
| `nltk` | Stopwords and stemming |
| `scikit-learn` | TF-IDF, Logistic Regression, accuracy |
| `pickle` | Model serialization |
| `kaggle` | Dataset download via API |

---

## 📊 Dataset

**Sentiment140** — [kaggle.com/datasets/kazanova/sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)

- 800,000 negative tweets (label `0`)
- 800,000 positive tweets (label `1`)
- Perfectly balanced — no class imbalance issues
- Labels derived automatically from emoticons (no manual annotation)

---

## 📈 Results

```
Training Accuracy  →  79.9%
Test Accuracy      →  77.7%
```

The ~2.2% gap between train and test accuracy shows the model generalises well without overfitting.
```
