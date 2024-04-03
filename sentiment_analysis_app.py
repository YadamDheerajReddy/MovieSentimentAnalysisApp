import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import numpy as np

# Load the tokenizer data
def load_tokenizer(tokenizer_path='tokenizer.json'):
    with open(tokenizer_path) as f:
        data = json.load(f)
        word_index = data['word_index']
        num_words = data['num_words']
        maxlen = data['maxlen']
    return word_index, num_words, maxlen

# Preprocess the text input
def preprocess_text(text, word_index, num_words, maxlen=100):
    # Convert words to word_indexes
    sequences = [[word_index[word] if word in word_index else 0 for word in text.split()]]
    # Ensure the sequence is within our num_words
    sequences = [[min(word, num_words - 1) for word in seq] for seq in sequences]
    # Pad sequences
    padded = pad_sequences(sequences, maxlen=maxlen)
    return padded

# Load model and tokenizer
model = load_model('sentiment_model.h5')
word_index, num_words, maxlen = load_tokenizer()

# Streamlit UI
st.title('IMDB Sentiment Analysis')

# User text input
user_input = st.text_area("Enter a movie review:")

if st.button('Predict Sentiment'):
    processed_input = preprocess_text(user_input, word_index, num_words, maxlen)
    prediction = model.predict(processed_input)
    sentiment = "Positive" if np.squeeze(prediction) > 0.5 else "Negative"
    
    st.write(f"Sentiment: {sentiment}")