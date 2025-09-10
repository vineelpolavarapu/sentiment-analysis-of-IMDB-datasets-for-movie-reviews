import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re

model=tf.keras.models.load_model("imdb_sentiment_analysis.h5")

with open("imdb_word_index.pkl","rb") as f:
    word_index=pickle.load(f)

with open("maxlen.pkl","rb") as f:
    maxlen=pickle.load(f)

# encode user input based on IMDB word index
def encode_text(text):
    text = re.sub(r'[^a-z0-9\s]', '', text.lower())
    token=text.split()
    encode=[word_index.get(word,2) for word in token]
    print(f"token :{token}")
    print(f"encode :{encode}")
    return encode

st.title("Movie reviews from IMDB data sets")
user_input=st.text_area("Enter your movie review" , height=150)

if st.button("Predict Sentiment"):
    if not user_input.strip():
        st.warning("Invalid input, please give the review")
    else:
        encoded_review = encode_text(user_input)
        padded_review = pad_sequences([encoded_review], maxlen=maxlen, padding="post", truncating="post")
        prediction = model.predict(padded_review)[0][0]
        sentiment = "Positive" if prediction > 0.5 else "Negative"
        st.success(f"Movie review is {sentiment} (Confidence: {prediction:.2f})")
