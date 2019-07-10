from typing import Optional, Tuple, Dict

import tensorflow as tf
from tensorflow.keras.layers import Reshape, Dense, Dropout, Flatten, Embedding, Conv1D, MaxPooling1D, Lambda, LSTM, GRU, CuDNNLSTM, Bidirectional, Input, Concatenate, GlobalMaxPooling1D
from tensorflow.keras.models import Model, Sequential

def lstm_chars_simple(input_shape: Tuple[int, ...], output_shape: Tuple[int, ...], hidden_units: int=128, num_layers: int=3, dropout: float=0.2)-> Model:
    num_classes = output_shape[0]

    inputs = Input(input_shape)
    X = Embedding(134, output_dim=4)(inputs)
    for _ in range(num_layers):
        X = CuDNNLSTM(hidden_units, return_sequences=True, go_backwards=True)(X)
    X = Dropout(dropout)(X)
    output = Dense(num_classes, activation='softmax')(X)

    return Model(inputs=inputs, outputs=output)
