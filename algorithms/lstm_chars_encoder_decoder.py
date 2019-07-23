from typing import Optional, Tuple, Dict

import tensorflow as tf
from tensorflow.keras.layers import Reshape, Dense, Dropout, Flatten, Embedding, Conv1D, MaxPooling1D, Lambda, LSTM, GRU, CuDNNLSTM, Bidirectional, Input, Concatenate, GlobalMaxPooling1D
from tensorflow.keras.layers import RepeatVector, Activation
from tensorflow.keras.models import Model, Sequential

repeator = RepeatVector(Tx)
concatenator = Concatenate(axis=-1)
densor1 = Dense(10, activation = "tanh")
densor2 = Dense(1, activation = "relu")
activator = Activation(softmax, name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
dotor = Dot(axes = 1)

def one_step_attention(a, s_prev):
    """
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.

    Arguments:
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)

    Returns:
    context -- context vector, input of the next (post-attetion) LSTM cell
    """

    # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a" (≈ 1 line)
    s_prev = repeator(s_prev)
    # Use concatenator to concatenate a and s_prev on the last axis (≈ 1 line)
    concat = concatenator([a, s_prev])
    # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e. (≈1 lines)
    e = densor1(concat)
    # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies. (≈1 lines)
    energies = densor2(e)
    # Use "activator" on "energies" to compute the attention weights "alphas" (≈ 1 line)
    alphas = activator(energies)
    # Use dotor together with "alphas" and "a" to compute the context vector to be given to the next (post-attention) LSTM-cell (≈ 1 line)
    context = dotor([alphas, a])

    return context


def lstm_chars_encoder_decoder(input_shape: Tuple[int, ...], output_shape: Tuple[int, ...],
            hidden_units_encoder: int=128, num_layers_encoder: int=3, hidden_units_decoder: int=128, num_layers_decoder: int=3, dropout: float=0.2)-> Model:
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    hidden_units_encoder -- hidden state size of the Bi-LSTM
    hidden_units_decoder -- hidden state size of the post-attention LSTM
    human_vocab_size -- size of the python dictionary "human_vocab"
    machine_vocab_size -- size of the python dictionary "machine_vocab"

    Returns:
    model -- Keras model instance
    """
    Tx = input_shape[0]
    Ty = output_shape[0]
    inputs = Input(input_shape)
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0
    #Initialize empty list of outputs
    outputs = []

    chars = Embedding(132, output_dim=4)(inputs)
    chars = Bidirectional(CuDNNLSTM(4, return_sequences=False))(chars)
    chars = Lambda(lambda x: tf.expand_dims(x, -1))(chars)
    for _ in range(num_layers_encoder):
        chars = Bidirectional(CuDNNLSTM(hidden_units_encoder, return_sequences=True))(chars)

    for t in range(Ty):
        context = one_step_attention(chars, s)
        s, _, c = CuDNNLSTM(hidden_units_decoder)(context, [s, c])
        #132 = machine_vocab_size
        out = Dense(132, activation='softmax')(s)
        outputs.append(out)

    return Model(inputs=[inputs, s0, c0], outputs=outputs)
