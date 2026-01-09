# Your-Smartest-Conversational-Partner---Sentiment-Analysis
This project performs **sentiment analysis on user reviews** using traditional Machine Learning techniques and presents results through an **interactive Streamlit web application** with detailed Exploratory Data Analysis (EDA).

## 🔍 Problem Statement

User reviews often contain opinions such as *good*, *bad*, or *average*, but the text data can be **short, noisy, and inconsistent**.  
The goal of this project is to:
- Classify reviews into **Positive / Negative / Neutral**
- Analyze trends across platforms, versions, and user behavior
- Build an end-to-end ML + UI solution

---

## 🧠 Models Used

The following models were trained and evaluated:
-**Logistic Regression**
-**LinearSVC**
- **Random Forest Classifier**


### ⚠️ Model Performance Note
Even after tuning:
- Random Forest achieved ~**58–66% accuracy**
- when compared to other models Random forest gives better accuracy


This is expected because:
- Reviews are **very short**
- Dataset contains **non-meaningful or ambiguous text**
- Sentiment classes overlap heavily

> ⚠️ Accuracy is limited by data quality, not model capability.

---

## 🛠️ Text Processing

- Lowercasing
- Punctuation removal
- Stopword removal
- TF-IDF Vectorization (used inside pipeline)
- Tokenization handled internally by TF-IDF  
---

## 📈 Exploratory Data Analysis (EDA)

The Streamlit app includes the following analyses:

1. **Overall sentiment distribution**
2. **Sentiment vs Rating mismatch**
3. **Keywords associated with each sentiment**
4. **Verified vs non-verified user sentiment**
5. **Review length vs sentiment**
6. **Sentiment across platforms (Web vs Mobile)**
7. **ChatGPT version vs sentiment / rating**
8. **Negative feedback theme identification**

All visualizations are rendered using **Matplotlib inside Streamlit**.

---

## 🌐 Streamlit Web App Features

- User enters a review → sentiment predicted
- Rule-based handling for very short inputs (e.g., *good*, *bad*)

- 
## 📦 Project Structure
── app.py # Streamlit application
├── model_pipeline.pkl # Saved ML pipeline (vectorizer + model)
├── requirements.txt # Required dependencies
├── README.md # Project documentation
└── data/
└── reviews.csv


📌 Technologies Used

Python
Pandas
Scikit-learn
Matplotlib
NLTK
Streamlit
Wordcloud

✅ Conclusion

Traditional ML models can struggle with low-quality textual data

Accuracy alone should not be the only evaluation metric

The project demonstrates complete ML workflow, EDA, and deployment


🔮 Future Improvements

Use Word Embeddings (Word2Vec / GloVe)

Apply LSTM / Transformer-based models

Improve dataset quality

Add sentiment confidence score


