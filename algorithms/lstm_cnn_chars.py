from typing import Optional, Tuple, Dict

import tensorflow as tf
from tensorflow.keras.layers import Reshape, Dense, Dropout, Flatten, Embedding, Conv1D, MaxPooling1D, Lambda, LSTM, GRU, CuDNNLSTM, Bidirectional, Input, Concatenate, GlobalMaxPooling1D
from tensorflow.keras.models import Model, Sequential

def lstm_cnn_chars(input_shape: Tuple[int, ...], output_shape: Tuple[int, ...], dropout: float=0.2)-> Model:
    num_classes = output_shape[0]

    inputs = Input(input_shape)
    charsEmbeddings = Embedding(600, output_dim=4)(inputs)
    chars1 = Conv1D(3, kernel_size=2, activation='tanh', input_shape=input_shape)(charsEmbeddings)
    chars1 = Flatten()(chars1)
    chars2 = Conv1D(4, kernel_size=3, activation='tanh', input_shape=input_shape)(charsEmbeddings)
    chars2 = Flatten()(chars2)
    chars3 = Conv1D(5, kernel_size=4, activation='tanh', input_shape=input_shape)(charsEmbeddings)
    chars3 = Flatten()(chars3)
    #chars4 = Conv1D(100, kernel_size=5, activation='relu', input_shape=input_shape)(charsEmbeddings)
    #chars4 = MaxPooling1D(pool_size=4)(chars4)
    #chars4 = Flatten()(chars4)
    chars = Concatenate(axis=-1)([chars1, chars2, chars3])
    chars = Lambda(lambda x: tf.expand_dims(x, -1))(chars)
    chars = GlobalMaxPooling1D()(chars)
    chars = Lambda(lambda x: tf.expand_dims(x, -1))(chars)
    X = CuDNNLSTM(30, return_sequences=False, go_backwards=True)(X)
    output = Dense(num_classes, activation='softmax')(X)

    return Model(inputs=inputs, outputs=output)
