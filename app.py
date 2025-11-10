import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import json
import numpy as np
import re

# =========================
# ---- Load Saved Files ----
# =========================
@st.cache_resource
def load_model_and_assets():
    model = tf.keras.models.load_model("fake_news_lstm_model.h5")

    # Load Tokenizer
    with open("fake_news_tokenizer.json", "r") as f:
        tokenizer_json = json.load(f)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)

    # Load Label Encoder
    label_encoder = joblib.load("label_encoder.pkl")

    return model, tokenizer, label_encoder

model, tokenizer, label_encoder = load_model_and_assets()

# =========================
# ---- Helper Functions ----
# =========================
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    text = text.split()
    return " ".join(text)

def predict_news(news_text):
    max_len = 300
    clean = clean_text(news_text)
    seq = tokenizer.texts_to_sequences([clean])
    pad = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    pred = (model.predict(pad) > 0.5).astype("int32")
    label = label_encoder.inverse_transform(pred)[0]
    return label

# =========================
# ---- Streamlit UI ----
# =========================
st.set_page_config(page_title="📰 Fake News Detection", layout="wide")

st.title("🧠 Fake News Detection using TensorFlow LSTM")
st.markdown("Enter any news headline or full article below to check whether it's **FAKE** or **REAL**.")

# Text Input
news_text = st.text_area("📰 Enter News Article Text:", height=200)

if st.button("🔍 Predict"):
    if news_text.strip() == "":
        st.warning("⚠️ Please enter some text first.")
    else:
        with st.spinner("Analyzing..."):
            label = predict_news(news_text)
            if label == "FAKE":
                st.error("🚨 This news appears to be **FAKE**!")
            else:
                st.success("✅ This news appears to be **REAL**!")

# Optional Footer
st.markdown("---")
st.markdown("Developed by **Arvind Sharma** | Powered by TensorFlow & Streamlit")
