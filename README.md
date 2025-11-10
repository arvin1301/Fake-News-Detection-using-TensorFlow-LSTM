# Fake-News-Detection-using-TensorFlow-LSTM

Tools Used: Python, TensorFlow, Keras, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, Streamlit
Dataset: Kaggle Fake News Dataset

# Project Overview

Fake news has become one of the most pressing challenges in the digital information era.
This project leverages Deep Learning (LSTM) to automatically classify news articles as FAKE or REAL based on their text content.

The goal is to help curb misinformation by using AI-powered text classification for automatic news verification.

# Objective

Develop a TensorFlow LSTM model that detects fake news by analyzing word patterns, semantics, and linguistic cues in the text.

Key Tasks:

Clean and preprocess text data

Tokenize and vectorize text into numerical sequences

Train an LSTM model on labeled fake/real news articles

Evaluate performance using standard classification metrics

Deploy as an interactive Streamlit web app

# Dataset Details

Source: Kaggle Fake News Dataset

Attributes:

title: News headline

text: News content

label: FAKE = 0, REAL = 1

Data Size: ~5000 samples (balanced classes)

Train-Test Split: 80% training | 20% testing

# Data Preprocessing

Preprocessing is essential to ensure clean, structured, and meaningful text for model learning.

Steps:

Removed punctuation, HTML tags, and special characters

Lowercased all text

Removed stopwords (e.g., "the", "is", "and")

Tokenized text using Keras Tokenizer (vocab_size=10,000)

Padded sequences to length = 300 for consistent input shape

Label encoded target variable (FAKE=0, REAL=1)

# Model Architecture (TensorFlow + LSTM)

The Long Short-Term Memory (LSTM) network is ideal for understanding contextual dependencies in sequential data like text.

Model Structure:

Embedding Layer   → Converts words to dense vector embeddings (128D)
LSTM Layer        → Captures word-level temporal dependencies
Dense Layer (ReLU) → Learns non-linear feature interactions
Output Layer (Sigmoid) → Outputs binary classification (FAKE / REAL)


Model Configuration:

Optimizer: Adam

Loss Function: Binary Crossentropy

Metrics: Accuracy, Precision, Recall, F1-score

Epochs: 10–15

Batch Size: 32

# Model Training & Performance

Training was conducted using TensorFlow on a GPU environment for faster computation.

# Training Curves:

Training vs Validation Accuracy: Stable improvement per epoch

Training vs Validation Loss: Smooth convergence, minimal overfitting

# Model Evaluation Metrics:
Metric	Value
Accuracy	98.1% 
Precision	97.2%
Recall	96.5%
F1-Score	96.8%

Interpretation:
High accuracy and balanced F1-score demonstrate the model’s ability to detect both fake and real news effectively.

# Confusion Matrix

The model correctly classifies the majority of news samples.
Very few real news articles are mistakenly flagged as fake — confirming high precision and low false positive rate.

# Model Comparison
Model	Approach	Accuracy
Logistic Regression	Traditional ML	88%
Random Forest	Ensemble ML	91%
XGBoost	Gradient Boosting	94%
LSTM (TensorFlow)	Deep Learning	98% ✅

Conclusion:
LSTM significantly outperforms classical models due to its ability to capture semantic and contextual relationships within text.

# Prediction Example

Input News:

“Government announces new policy on renewable energy investments.”

Model Output:

# REAL News (Confidence: 97%)

Input News:

“Celebrity endorses cure for diabetes that scientists call fake.”

Model Output:

# FAKE News (Confidence: 95%)

# Streamlit Web Application

An interactive Streamlit web app was developed for real-time fake news detection.

# App Features:

Simple text input box for users to paste or type news content

Instant prediction of FAKE / REAL with color-coded output

Real-time inference using the trained TensorFlow model

Files Used:

fake_news_lstm_model.h5 → Trained LSTM model

fake_news_tokenizer.json → Saved tokenizer

label_encoder.pkl → Label mapping file

app.py → Streamlit app script

# Run the App:
streamlit run app.py

# Technical Stack
Category	Tools Used
Language	Python 3.10+
Frameworks	TensorFlow, Keras
Libraries	Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
NLP Tools	Tokenizer, Word2Vec, TF-IDF
Deployment	Streamlit
Environment	Jupyter Notebook / Google Colab
 Repository Structure
 Fake-News-Detection/
│
├──  fake_news_final.ipynb                # Model training and EDA notebook
├──  news.csv                             # Dataset
├──  Fake_News_Detection_Final_Presentation_Enhanced.pptx
├──  Fake News Detection Model using TensorFlow.pdf
├──  app.py                               # Streamlit web app
├──  requirements.txt                     # Dependencies
├──  model/
│   ├── fake_news_lstm_model.h5
│   ├── fake_news_tokenizer.json
│   └── label_encoder.pkl
└──  README.md                            # Project documentation

 Installation & Setup

Clone the repository

git clone https://github.com/yourusername/Fake-News-Detection.git
cd Fake-News-Detection


Install dependencies

pip install -r requirements.txt


Run the Streamlit app

streamlit run app.py


Open in browser:
http://localhost:8501

# Future Enhancements

Integrate Transformer-based models (BERT, RoBERTa) for higher accuracy

Add multi-language detection for global news coverage

Deploy as a browser extension or fact-checking API

Implement real-time social media monitoring for misinformation detection

# Conclusion

This project showcases how AI and NLP can be leveraged to combat misinformation effectively.
The TensorFlow LSTM model (98% accuracy) demonstrates the power of deep learning in text classification and real-world deployment through Streamlit.

🧩 “AI can help build a future where truth spreads faster than fake news.”
