import streamlit as st
import joblib
import re
import string

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# App title
st.title("📰 Fake News Detector")
st.write("Enter news below to detect if it's **Real** or **Fake**.")

# Input
news_text = st.text_area("News text:")

# Clean the text
def clean_text(text):
    return re.sub(f"[{string.punctuation}]", "", text.lower())

# Predict button
if st.button("Predict"):
    if news_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(news_text)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        if prediction == 1:
            st.success("✅ This looks like **Real News**")
        else:
            st.error("⚠️ This looks like **Fake News**")
