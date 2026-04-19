# seq2seq

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample dataset
input_texts = ["hi", "hello", "how are you"]
target_texts = ["hola", "hola", "como estas"]

# Tokenization
input_tokenizer = Tokenizer()
target_tokenizer = Tokenizer()

input_tokenizer.fit_on_texts(input_texts)
target_tokenizer.fit_on_texts(target_texts)

input_seq = input_tokenizer.texts_to_sequences(input_texts)
target_seq = target_tokenizer.texts_to_sequences(target_texts)

input_seq = pad_sequences(input_seq, padding='post')
target_seq = pad_sequences(target_seq, padding='post')

# Vocabulary sizes
input_vocab_size = len(input_tokenizer.word_index) + 1
target_vocab_size = len(target_tokenizer.word_index) + 1

latent_dim = 64

# ---------------- ENCODER ----------------
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(input_vocab_size, latent_dim)(encoder_inputs)

encoder_lstm = LSTM(latent_dim, return_state=True)
_, state_h, state_c = encoder_lstm(encoder_embedding)

encoder_states = [state_h, state_c]

# ---------------- DECODER ----------------
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(target_vocab_size, latent_dim)(decoder_inputs)

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

decoder_dense = Dense(target_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Reshape target
target_seq = np.expand_dims(target_seq, -1)

# Train
model.fit([input_seq, target_seq], target_seq, epochs=50)

print("Training complete")

# Predict on training data
pred = model.predict([input_seq, target_seq])

# Reverse word index
reverse_target_word_index = {v: k for k, v in target_tokenizer.word_index.items()}

def decode_sequence(seq):
    return ' '.join([reverse_target_word_index.get(i, '') for i in seq if i != 0])

for i, p in enumerate(pred):
    predicted_sequence = np.argmax(p, axis=-1)
    
    print("Input:", input_texts[i])
    print("Predicted:", decode_sequence(predicted_sequence))
    print("Actual:", target_texts[i])
    print()