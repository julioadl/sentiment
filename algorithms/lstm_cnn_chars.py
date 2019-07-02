from typing import Optional, Tuple, Dict

import tensorflow as tf
from tensorflow.keras.layers import Reshape, Dense, Dropout, Flatten, Embedding, Conv1D, MaxPooling1D, Lambda, LSTM, GRU, CuDNNLSTM, Bidirectional, Input, Concatenate, GlobalMaxPooling1D
from tensorflow.keras.models import Model, Sequential

def lstm_cnn_chars(input_shape: Tuple[int, ...], output_shape: Tuple[int, ...], dropout: float=0.2)-> Model:
    num_classes = output_shape[0]

    inputs = Input(input_shape)
    #8700 as it is 29 words x 300 the dimension of the embeddings
#    charsInputs = Lambda(lambda x: x[:,:240])(inputs)
    charsEmbeddings = Embedding(193, output_dim=4)(inputs)
    chars1 = Conv1D(3, kernel_size=2, activation='tanh', input_shape=input_shape, kernel_initializer='glorot_normal')(charsEmbeddings)
    chars1 = Flatten()(chars1)
    chars2 = Conv1D(4, kernel_size=3, activation='tanh', input_shape=input_shape, kernel_initializer='glorot_normal')(charsEmbeddings)
    chars2 = Flatten()(chars2)
    chars3 = Conv1D(5, kernel_size=4, activation='tanh', input_shape=input_shape, kernel_initializer='glorot_normal')(charsEmbeddings)
    chars3 = Flatten()(chars3)
    chars = Concatenate(axis=-1)([chars1, chars2, chars3])
    chars = Lambda(lambda x: tf.expand_dims(x, -1))(chars)
    chars = MaxPooling1D()(chars)
    chars = BatchNormalization()(chars)
    chars = Bidirectional(CuDNNLSTM(8, return_sequences=True, kernel_initializer='glorot_normal'))(chars)
    chars = Bidirectional(CuDNNLSTM(8, return_sequences=True, kernel_initializer='glorot_normal'))(chars)
    chars = Bidirectional(CuDNNLSTM(8, return_sequences=True, kernel_initializer='glorot_normal'))(chars)
    chars = Flatten()(chars)
    X = Dropout(dropout)(chars)
    output = Dense(num_classes, activation='sigmoid', kernel_initializer='random_normal')(chars)

    return Model(inputs=inputs, outputs=output)
