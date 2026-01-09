import streamlit as st
import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud


st.set_page_config(page_title="Sentiment Analysis App", layout="wide")

st.title("📊 Sentiment Analysis & EDA Dashboard")

# Load pipeline
pipeline = pickle.load(open("sentiment_pipeline.pkl", "rb"))

# Load dataset
df = pd.read_csv("chatgpt_style_reviews_dataset.csv")

def get_sentiment(rating):
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"
    


df['sentiment'] = df['rating'].apply(get_sentiment)

st.header("🔍 Sentiment Predictor")

user_input = st.text_area("Enter a review:")

if st.button("Analyze Sentiment"):
    text = user_input.lower().strip()


    positive_words = ["good", "great", "excellent", "awesome", "nice", "amazing","well"]
    negative_words = ["bad", "worst", "poor", "terrible", "awful", "hate","not"]

    if text in positive_words:
        sentiment = "Positive"
        st.success(f"Predicted Sentiment: {sentiment}")

    elif text in negative_words:
        sentiment = "Negative"
        st.error(f"Predicted Sentiment: {sentiment}")

    else:
        prediction = pipeline.predict([user_input])[0]
        label_map = {
            0: "Negative",
            1 : "Positive"
        }

        sentiment = label_map[prediction]

        if sentiment == "Positive":
            st.success(f"Predicted Sentiment: {sentiment}")
        else:
            st.error(f"Predicted Sentiment: {sentiment}")

# EDA Analysis
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Overall Sentiment",
    "Sentiment vs Rating",
    "Keywords",
    "Verified Users",
    "Review Length",
    "Average Rating by Platform",
    "Average Rating by ChatGPT Major Version"
])


with tab1:
    st.subheader("Overall Sentiment Distribution")

    sentiment_counts = df['sentiment'].value_counts(normalize=True) * 100
    col1, col2 = st.columns([1, 2])  # narrow + wide

    with col1:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(sentiment_counts.index, sentiment_counts.values,
               color=['red', 'green', 'gray'])
        ax.set_ylabel("Percentage")
        st.pyplot(fig)


nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [word for word in text.split() if word not in stop_words]
    return " ".join(words)

df['clean_review'] = df['review'].apply(preprocess_text)



with tab2:
    st.subheader("Sentiment vs Star Rating")

    col1, col2 = st.columns([1, 2]) 

    with col1:
        rating_sentiment = pd.crosstab(df['rating'], df['sentiment'], normalize='index')

        st.dataframe(rating_sentiment)

        fig, ax = plt.subplots()
        rating_sentiment.plot(kind='bar', stacked=True, ax=ax)
        st.pyplot(fig)


with tab3:
    st.subheader("Keywords by Sentiment")

    col1, col2 = st.columns([1, 2]) 

    with col1:
        sentiment_choice = st.selectbox(
        "Choose sentiment",
        ["Positive", "Neutral", "Negative"]
        )

        text = " ".join(df[df['sentiment'] == sentiment_choice]['clean_review'])

        wordcloud = WordCloud(
             width=800,
             height=400,
             background_color='white'
        ).generate(text)

        fig, ax = plt.subplots()
        ax.imshow(wordcloud)
        ax.axis("off")

        st.pyplot(fig)

with tab4:
    st.subheader("Verified Purchase vs Sentiment")
    col1, col2 = st.columns([1, 2]) 

    with col1:
        verified_sentiment = pd.crosstab(
        df['verified_purchase'],
        df['sentiment'],
        normalize='index'
        )

        fig, ax = plt.subplots()
        verified_sentiment.plot(kind='bar', ax=ax)
        st.pyplot(fig)

with tab5:
    st.subheader("Review Length vs Sentiment")
    col1, col2 = st.columns([1, 2]) 

    with col1:
        df['review_length'] = df['clean_review'].str.len()

        avg_length = df.groupby('sentiment')['review_length'].mean()

        fig, ax = plt.subplots()
        ax.bar(avg_length.index, avg_length.values,
           color=['red', 'gray', 'green'])

        ax.set_ylabel("Average Review Length")

        st.pyplot(fig)
with tab6:
    st.subheader("Average Rating by Platform")
    col1, col2 = st.columns([1, 2])  

    with col1:
        platform_avg_rating = df.groupby('platform')['rating'].mean().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(6, 4))  # Smaller figure for Streamlit
        ax.bar(platform_avg_rating.index, platform_avg_rating.values, color='orange')
        ax.set_xlabel("Platform")
        ax.set_ylabel("Average Rating")
        ax.set_title("Average Rating by Platform")
        plt.xticks(rotation=30)  # Optional

        st.pyplot(fig, use_container_width=False)  

with tab7:
    st.subheader("Average Rating by ChatGPT Major Version")
    col1, col2 = st.columns([1, 2])  # narrow + wide

    with col1:
        df['major_version'] = df['version'].str.split('.').str[0]

        # Compute average rating per major version
        avg_rating_major = df.groupby('major_version')['rating'].mean().sort_index()


        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(avg_rating_major.index, avg_rating_major.values, color='skyblue')
        ax.set_xlabel("ChatGPT Major Version")
        ax.set_ylabel("Average Rating")
        ax.set_title("Average Rating by Major Version")
        plt.xticks(rotation=30)  

        st.pyplot(fig, use_container_width=False)
