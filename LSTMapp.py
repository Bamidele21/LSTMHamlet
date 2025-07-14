##-----------Next Word Prediction using LSTM with StreamLit----------------
#-- Importing Libraries
import tensorflow as tf
import numpy as np
import pandas as pd
import nltk
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping




import streamlit as st

#-- Data Collection
# ---Data Preproccessing
with open('hamlet.txt', 'r') as file:
    text = file.read().lower()
    
# ---Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
vocab_size = len(tokenizer.word_index) + 1
#print (f"Vocabulary Size: {vocab_size}")

# ---Creating Sequences
sequences = []
for line in text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        ng_seq = token_list[:i + 1]
        sequences.append(ng_seq)
        
#print (sequences)
# ---Padding Sequences
max_seq_len = max([len(x) for x in sequences])

sequences = np.array(pad_sequences(sequences, maxlen=max_seq_len, padding='pre'))
        
##print(sequences)
### ---Splitting Data
X, y = sequences[:, :-1], sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Define EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss',  patience=3, restore_best_weights=True)
# ---Model Creation
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_seq_len-1 ))
model.add(LSTM(150, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(vocab_size, activation='softmax'))
model.build(input_shape=(None, max_seq_len))

# ---Model Compilation
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
##model.summary()

# ---Model Training
##model.fit(x_train, y_train, epochs=200, verbose=1, validation_data=(x_test, y_test))

#model.save('hamlet_next_word_model.h5')

# Load the model
@st.cache_resource
def load_model_from_file():
    model = load_model('hamlet_next_word_model.h5')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = load_model_from_file()



#function to predict the next word
def predict_next_word(model, tokenizer, text, max_seq_len): 
    token_list = tokenizer.texts_to_sequences([text])[0]
    if not token_list:
        return "Input text is empty or not in vocabulary."
    if len(token_list) >= max_seq_len:
        token_list = token_list[-(max_seq_len-1):]  # ensures the sequence is at most max_seq_len-1 tokens, matching model input length
    token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=-1)[0]
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None
    
    
    
# Streamlit app
st.title("Next Word Prediction using LSTM")
input_text = st.text_input("Enter a sentence", "To be, or not to be that is")
if st.button("Predict"):
    max_seq_len = model.input_shape[1]+1  # Get the max sequence length from the model input shape
    next_word = predict_next_word(model, tokenizer, input_text, max_seq_len)
    st.write(f"The next word is: {next_word}")