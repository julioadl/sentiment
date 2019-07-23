from typing import Optional, Tuple, Dict

import tensorflow as tf
from tensorflow.keras.layers import Reshape, Dense, Dropout, Flatten, Embedding, Conv1D, MaxPooling1D, Lambda, LSTM, GRU, CuDNNLSTM, Bidirectional, Input, Concatenate, GlobalMaxPooling1D
from tensorflow.keras.models import Model, Sequential

def lstm_chars(input_shape: Tuple[int, ...], output_shape: Tuple[int, ...], hidden_units: int=128, num_layers: int=3, dropout: float=0.2)-> Model:
    num_classes = output_shape[0]
    inputs = Input(input_shape)
    #8700 as it is 29 words x 300 the dimension of the embeddings
    #chars = Lambda(lambda x: x[:,:240])(inputs)
    chars = Embedding(132, output_dim=4)(inputs)
    chars = Bidirectional(CuDNNLSTM(4, return_sequences=False))(chars)
    chars = Lambda(lambda x: tf.expand_dims(x, -1))(chars)
    for _ in range(num_layers):
        chars = Bidirectional(CuDNNLSTM(hidden_units, return_sequences=True))(chars)
    chars = Flatten()(chars)
    X = Dropout(dropout)(chars)
    output = Dense(num_classes, activation='sigmoid')(X)

    return Model(inputs=inputs, outputs=output)
