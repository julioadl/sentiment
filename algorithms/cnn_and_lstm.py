'''
Experiment based on insights from this paper:

https://arxiv.org/pdf/1702.01923.pdf
'''
from typing import Optional, Tuple, Dict

import tensorflow as tf
from tensorflow.keras.layers import Reshape, Dense, Dropout, Flatten, Embedding, Conv1D, MaxPooling1D, Lambda, LSTM, GRU, CuDNNLSTM, Bidirectional, Input, Concatenate, GlobalMaxPooling1D
from tensorflow.keras.models import Model, Sequential

def cnn_and_lstm(input_shape: Tuple[int, ...], output_shape: Tuple[int, ...], dropout: float=0.2)-> Model:
    num_classes = output_shape[0]

    inputs = Input(input_shape)
    charsEmbeddings = Embedding(194, output_dim=4)(inputs)
    #CNN architecture
    chars1 = Conv1D(3, kernel_size=2, activation='tanh', input_shape=input_shape)(charsEmbeddings)
    chars1 = Flatten()(chars1)
    chars2 = Conv1D(4, kernel_size=3, activation='tanh', input_shape=input_shape)(charsEmbeddings)
    chars2 = Flatten()(chars2)
    chars3 = Conv1D(5, kernel_size=4, activation='tanh', input_shape=input_shape)(charsEmbeddings)
    chars3 = Flatten()(chars3)
    words = Concatenate(axis=-1)([chars1, chars2, chars3])
    words = Lambda(lambda x: tf.expand_dims(x, -1))(words)
    words = MaxPooling1D()(words)
    words = Flatten()(words)

    #LSTM architecture
    lstm = CuDNNLSTM(121, return_sequences=False, go_backwards=True)(charsEmbeddings)

    #Concatenate
    X = Concatenate(axis=-1)([lstm, words])
    X = Dropout(dropout)(X)
    X = Dense(30, activation='relu')(X)
    X = Dropout(dropout)(X)
    output = Dense(num_classes, activation='softmax')(X)

    return Model(inputs=inputs, outputs=output)
