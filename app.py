import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

word_indexes = imdb.get_word_index()

reverse_dict = dict()

for key, value in word_indexes.items():
    reverse_dict[value] = key

model = load_model('artifacts/simplernn.keras')

def preprocess(text:str):
    words = text.lower().split()
    encoded_word = []
    encoded_word = [word_indexes.get(word, 2) + 3 for word in words]
    padded_word = pad_sequences([encoded_word], 500)

    return padded_word

def predict(text):
    preprocessed_input = preprocess(text)
    
    prediction = model.predict(preprocessed_input)[0][0]

    return prediction, 'Positive' if prediction >= 0.5 else 'Negative'

st.title("IMDB Movie Review Sentiment Analysis")
st.write('Enter a movie review for classification')
user_input = st.text_area('Review')

if st.button('Classify'):
    probability, sentiment = predict(user_input)

    st.write(f'The Predicted Sentiment: {sentiment}')
    st.write(f'The Probability of the Sentiment: {probability * 100:.2f}%')

else:
    st.write("Please enter a review")