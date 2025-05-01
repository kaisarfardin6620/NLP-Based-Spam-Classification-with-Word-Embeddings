# NLP-Based-Spam-Classification-with-Word-Embeddings

This project focuses on detecting spam emails using natural language processing techniques combined with pretrained word embeddings such as Glov, FastText and Word2Vec. It applies both traditional machine learning models and lays the groundwork for deep learning approaches (e.g., LSTM/GRU).

## 📌 Problem Statement

The goal is to classify email messages into two categories:
- **Spam**
- **Not Spam (Ham)**

The classification is based on the textual content of the email, using vectorized representations from pretrained embedding models.

## 🧠 Techniques Used

### 🔹 Preprocessing
- Lowercasing and punctuation removal
- Tokenization and cleaning
- Handling empty or null values

### 🔹 Feature Extraction
- Pretrained word embeddings (Word2Vec, Glov, FastText via Gensim)
- Sentence vectorization by averaging word embeddings

### 🔹 Modeling
- Traditional classifiers:
  - Logistic Regression
  - XGBoost
  - Support Vector Machine (SVM)
- Preparation for neural models like LSTM/GRU

## 📁 Dataset

The dataset contains email texts labeled as spam or ham. It is assumed to have the following columns:

- `text`: The email message content
- `label`: 0 (ham) or 1 (spam)

> The dataset can be loaded from a CSV file and preprocessed for training and evaluation.

## 📊 Evaluation Metrics

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**

## 📦 Libraries Used

- `pandas`, `numpy`
- `scikit-learn`
- `xgboost`
- `gensim`
- `matplotlib`, `seaborn`

## 🚀 Future Work

- Use sequence-based input for LSTM/GRU models
- Experiment with transformer models like BERT or DistilBERT
- Add web-based interface or email classifier API
- Include hyperparameter tuning and cross-validation
