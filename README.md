# 🐦 Twitter Sentiment Analysis
### Classifying tweets as Positive or Negative using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-1.6M%20Tweets-blue?style=flat-square)
![Accuracy](https://img.shields.io/badge/Accuracy-~79%25-brightgreen?style=flat-square)

---

<p align="center">
  <img src="https://images.unsplash.com/photo-1611605698335-8b1569810432?w=1200&h=400&fit=crop&crop=center" width="100%" alt="Twitter Social Media Banner"/>
</p>

---

## 📌 Overview

This project builds a **sentiment classifier** that reads a tweet and predicts whether it's **positive** or **negative**. It uses the [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) with 1.6 million labeled tweets, trained with Logistic Regression and TF-IDF features.

---

## 🔄 Pipeline at a Glance

```
Raw Tweets (CSV)  →  Preprocessing  →  Stemming  →  TF-IDF  →  Logistic Regression  →  Prediction
```

<p align="center">
  <img src="https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=1200&h=420&fit=crop&crop=center" width="100%" alt="Data Analytics Pipeline"/>
</p>

---

## 📂 Dataset

- **Source:** [Kaggle – Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)
- **Size:** 1,600,000 tweets
- **Columns:** `sentiment`, `id`, `date`, `query`, `user`, `tweet`
- **Labels:** `0` → Negative &nbsp;|&nbsp; `1` → Positive (originally `4`, converted to `1`)

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| `pandas` / `numpy` | Data loading & manipulation |
| `NLTK` | Stopword removal & stemming |
| `scikit-learn` | TF-IDF vectorizer + Logistic Regression |
| `pickle` | Saving & loading the trained model |
| `Kaggle API` | Dataset download |

---

## ⚙️ How It Works

<p align="center">
  <img src="https://images.unsplash.com/photo-1504868584819-f8e8b4b6d7e3?w=1200&h=400&fit=crop&crop=center" width="100%" alt="Machine Learning Workflow"/>
</p>

### 1. Data Preprocessing
- Loaded the raw CSV with correct column names
- Checked for missing values (none found)
- Replaced sentiment label `4` → `1` for binary classification

### 2. Text Cleaning (Stemming)
Each tweet is cleaned using a custom `stemming()` function:
- Removes non-alphabetic characters
- Converts to lowercase
- Removes English stopwords
- Applies Porter Stemming to reduce words to their root form

### 3. Feature Extraction
- **TF-IDF Vectorizer** converts cleaned text into numerical feature vectors
- Trained only on the training set to avoid data leakage

### 4. Model Training
- **Logistic Regression** with `max_iter=1000`
- 80/20 train-test split with stratification to maintain class balance

### 5. Evaluation

| Split | Accuracy |
|-------|----------|
| Training | ~79% |
| Testing  | ~77% |

### 6. Model Saving
- Trained model saved as `trained_model.sav` using `pickle`
- Can be reloaded and used for new predictions anytime

---

## 🚀 Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/jathinreddy-5/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis

# 2. Install dependencies
pip install numpy pandas nltk scikit-learn kaggle

# 3. Set up Kaggle API
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 4. Download dataset
kaggle datasets download -d kazanova/sentiment140

# 5. Run the notebook
jupyter notebook Twitter_sentiment.ipynb
```

---

## 🔮 Making a Prediction

```python
import pickle

# Load saved model & vectorizer
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# Predict on a new tweet
tweet_vector = vectorizer.transform(["I love this product!"])
prediction = loaded_model.predict(tweet_vector)

print("Positive 😊" if prediction[0] == 1 else "Negative 😞")
```

---

## 📁 Project Structure

```
twitter-sentiment-analysis/
│
├── Twitter_sentiment.ipynb   # Main notebook
├── trained_model.sav         # Saved Logistic Regression model
├── kaggle.json               # Kaggle API credentials (do not share)
└── README.md
```

---

<p align="center">
  <img src="https://images.unsplash.com/photo-1518186285589-2f7649de83e0?w=1200&h=350&fit=crop&crop=center" width="100%" alt="Code and Analytics"/>
</p>

---



> ⭐ If you found this helpful, consider giving the repo a star!
