from typing import Optional, Tuple, Dict

import tensorflow as tf
from tensorflow.keras.layers import Reshape, Dense, Dropout, Flatten, Embedding, Conv1D, MaxPooling1D, Lambda, LSTM, GRU, CuDNNLSTM, Bidirectional, Input, Concatenate, GlobalMaxPooling1D
from tensorflow.keras.models import Model, Sequential

def lstm_chars_simple(input_shape: Tuple[int, ...], output_shape: Tuple[int, ...], dropout: float=0.2)-> Model:
    num_classes = output_shape[0]

    inputs = Input(input_shape)
    #charsEmbeddings = Embedding(155, output_dim=4)(inputs)
    #words = Lambda(lambda x: tf.expand_dims(x, -1))(charsEmbeddings)
    X = CuDNNLSTM(121, return_sequences=False, go_backwards=True)(inputs)
    X = Dropout(dropout)(X)
    #X = Dense(30, activation='relu')(X)
    #X = Dropout(dropout)(X)
    output = Dense(num_classes, activation='softmax')(X)

    return Model(inputs=inputs, outputs=output)
