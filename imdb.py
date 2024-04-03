import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import os

# Parameters
num_words = 10000  # Number of words to consider as features
maxlen = 100  # Cut texts after this number of words

# Load the IMDB dataset
print("Loading data...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

# Pad sequences for uniform input size
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# Define the model
model = Sequential([
    Embedding(num_words, 32),
    LSTM(32),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
print("Training model...")
model.fit(x_train, y_train, epochs=5, batch_size=256, validation_split=0.2)

# Save the model
model_save_path = 'sentiment_model.h5'
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

# Save the tokenizer
tokenizer_path = 'tokenizer.json'
word_index = imdb.get_word_index()
tokenizer_data = {'word_index': word_index, 'num_words': num_words, 'maxlen': maxlen}
with open(tokenizer_path, 'w', encoding='utf-8') as f:
    json.dump(tokenizer_data, f)
print(f"Tokenizer saved to {tokenizer_path}")
