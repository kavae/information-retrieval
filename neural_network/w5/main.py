#Avaya Khatri

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dense
from sklearn.metrics import confusion_matrix, precision_score, recall_score

# Dataset
transcripts = [
    "Revenues collapsed in Europe, however our Asian expansion broke all time records.",
    "Despite a massive surge in initial preorders, heavy supply chain failures ruined the quarter.",
    "We lost our biggest client, but the new merger will completely offset those losses.",
    "Profits are at an all time high, yet we face severe regulatory fines next month.",
    "The new product launch failed entirely, forcing us to rethink our strategy.",
    "Although inflation increased our overhead, strong consumer demand drove record profits."
]

labels = np.array([1, 0, 1, 0, 0, 1])

# Parameters
VOCAB_SIZE = 200
MAX_LENGTH = 15

# Tokenization
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(transcripts)
sequences = tokenizer.texts_to_sequences(transcripts)
padded_data = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post')

# Model
model = Sequential()
model.add(Input(shape=(MAX_LENGTH,)))
model.add(Embedding(input_dim=VOCAB_SIZE, output_dim=16))
model.add(Bidirectional(LSTM(units=16)))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("BiLSTM Architecture")
model.summary()

# Training
print("Training")
model.fit(padded_data, labels, epochs=25, verbose=0)

# Predictions on training data
pred_probs = model.predict(padded_data)
pred_labels = (pred_probs > 0.5).astype(int)

# Evaluation
cm = confusion_matrix(labels, pred_labels)
precision = precision_score(labels, pred_labels)
recall = recall_score(labels, pred_labels)

print("Evaluation")
print("Confusion Matrix:\n", cm)
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))

# Test sentence
test_sentence = ["We experienced a catastrophic drop in retail sales, though our new cloud software division completely saved our profit margins."]
test_seq = tokenizer.texts_to_sequences(test_sentence)
test_padded = pad_sequences(test_seq, maxlen=MAX_LENGTH, padding='post')

prediction = model.predict(test_padded)[0][0]

print("Test Prediction")
print(f"Sentence: {test_sentence[0]}")
print(f"Sentiment Score: {prediction:.4f}")
print("Result: UPGRADE" if prediction > 0.5 else "Result: DOWNGRADE")