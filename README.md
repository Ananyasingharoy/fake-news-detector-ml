# 📰 Fake News Detector (Machine Learning)

This project uses NLP and machine learning to classify news articles as **Real** or **Fake**.

##  Overview

- Dataset: [Kaggle – Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- Model: Logistic Regression
- Features: TF-IDF Vectorization
- Accuracy: ~98.41%

##  Predict Function Example

```python
import joblib, re, string

model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def predict_news(news_text):
    cleaned = re.sub(f"[{string.punctuation}]", "", news_text.lower())
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    return "Real News" if pred == 1 else "Fake News"
