#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Memory module for literature generating model using word embeddings."""


import numpy as np
import tensorflow as tf


# Shortcut for activation functions:
tanh = tf.keras.activations.tanh
sigmoid = tf.keras.activations.sigmoid
softmax = tf.keras.activations.softmax
relu = tf.keras.activations.relu


class MeLa(tf.keras.layers.Layer):
    """
    Memory module for literature generating model.

    At the core of the module there is a memory matrix. A single weight vector
    is used to control reading, erasing, and writing to/from the memory matrix.
    Separate recurrent layers control the content of the erase and write
    vectors. The text (input) is encoded by means of word-level embedding
    (word2vec).
    """

    def __init__(self,  # noqa
                 batch_size=None,
                 emb_size=None,
                 units_01=None,
                 mem_locations=None,
                 mem_size=None,
                 drop_in=None,
                 drop_state=None,
                 drop_mem=None,
                 name='LiGeMo',
                 **kwargs):
        super(MeLa, self).__init__(name=name, **kwargs)

        # ** Model parameters **
        self.lgm_batch_size = batch_size
        self.lgm_mem_locations = mem_locations
        self.lgm_mem_size = mem_size

        # ** Layers **

        # Input feedforward module:
        self.drop1 = tf.keras.layers.Dropout(drop_in)
        self.d1 = tf.keras.layers.Dense(units_01,
                                        input_shape=(batch_size,
                                                     1,
                                                     emb_size),
                                        activation=sigmoid,
                                        name='dense_in')

        # Layer controlling weight vector:
        self.mem_weights = tf.keras.layers.Dense(mem_locations,
                                                 activation=softmax,
                                                 name='memory_weights')

        # Layer controlling content of write vector:
        self.write = tf.keras.layers.Dense(mem_size,
                                           activation=sigmoid,
                                           name='memory_write')

        # Layer controlling content of erase vector:
        self.erase = tf.keras.layers.Dense(mem_size,
                                           activation=sigmoid,
                                           name='memory_erase')

        # Output feedforward module:
        self.drop2 = tf.keras.layers.Dropout(drop_in)
        self.d2 = tf.keras.layers.Dense(emb_size,
                                        activation=tanh,
                                        name='dense_out')

        # ** Dropouts **

        # Dropout layers for state arrays and memory:
        self.drop_memory = tf.keras.layers.Dropout(drop_mem)
        self.drop_state_weights = tf.keras.layers.Dropout(drop_state)
        self.drop_state_erase = tf.keras.layers.Dropout(drop_state)
        self.drop_state_write = tf.keras.layers.Dropout(drop_state)

        # ** Recurrent states **

        # The three layers (controlling the weights, erase, and write vectors)
        # have a recurrent state vector.

        # State of weights vector:
        vec_rand_01 = tf.random.normal((batch_size, mem_locations),
                                       mean=0.0,
                                       stddev=0.1,
                                       dtype=tf.float32)
        self.state_weights = tf.Variable(initial_value=vec_rand_01,
                                         trainable=False,
                                         dtype=tf.float32,
                                         name='state_weights')

        # State of erase vector:
        vec_rand_02 = tf.random.normal((batch_size, mem_size),
                                       mean=0.0,
                                       stddev=0.1,
                                       dtype=tf.float32)
        self.state_erase = tf.Variable(initial_value=vec_rand_02,
                                       trainable=False,
                                       dtype=tf.float32,
                                       name='state_erase')

        # State of write vector:
        vec_rand_03 = tf.random.normal((batch_size, mem_size),
                                       mean=0.0,
                                       stddev=0.1,
                                       dtype=tf.float32)
        self.state_write = tf.Variable(initial_value=vec_rand_03,
                                       trainable=False,
                                       dtype=tf.float32,
                                       name='state_write')

        # ** Memory **

        # Random values for initial state of memory vector:
        vec_rand_04 = tf.random.normal((batch_size, mem_locations, mem_size),
                                       mean=0.0,
                                       stddev=0.1,
                                       dtype=tf.float32)

        # The actual memory matrix:
        self.memory = tf.Variable(initial_value=vec_rand_04,
                                  trainable=False,
                                  dtype=tf.float32,
                                  name='memory_matrix')

        # Matrix of ones (needed for memory update):
        self.ones = tf.ones((batch_size, mem_locations, mem_size),
                            dtype=tf.float32,
                            name='ones_matrix')

        # Math ops:
        # self.add = tf.keras.layers.Add()
        # self.sub = tf.keras.layers.Subtract()
        # self.mult = tf.keras.layers.Multiply()
        self.conc = tf.keras.layers.Concatenate(axis=1)

        # Reshape op for output (batch size gets added automatically as first
        # dimension):
        self.reshape = tf.keras.layers.Reshape((1, emb_size))


    def call(self, inputs):  # noqa

        # Activate of first feedforward module:
        f1 = self.drop1(inputs)
        f1 = self.d1(f1)

        # Activate recurrent layer that controls weights vector:
        conc_01 = self.conc([f1[:, 0, :],
                            self.state_weights,   # batch_size * mem_locations
                            self.state_erase,     # batch_size * mem_size
                            self.state_write])    # batch_size * mem_size
        weight_vec = self.mem_weights(conc_01)

        # Activate recurrent layer that controls write vector:
        conc_02 = self.conc([f1[:, 0, :],
                            self.state_weights,   # batch_size * mem_locations
                            self.state_erase,     # batch_size * mem_size
                            self.state_write])    # batch_size * mem_size
        write_vec = self.write(conc_02)

        # Activate recurrent layer that controls erase vector:
        conc_03 = self.conc([f1[:, 0, :],
                            self.state_weights,   # batch_size * mem_locations
                            self.state_erase,     # batch_size * mem_size
                            self.state_write])    # batch_size * mem_size
        erase_vec = self.erase(conc_03)

        # Calculate read vector:
        read_vec = tf.linalg.matvec(self.memory,  # batch_size * mem_locations * mem_size
                                    weight_vec,   # batch_size * mem_locations
                                    transpose_a=True)

        # ** Calculate new memory matrix **

        # Multiply weight vector with erase vector:
        erase_mat = tf.linalg.matmul(tf.reshape(weight_vec,
                                                (self.lgm_batch_size,
                                                 self.lgm_mem_locations,
                                                 1)),
                                     tf.reshape(erase_vec,
                                                (self.lgm_batch_size,
                                                 1,
                                                 self.lgm_mem_size)),
                                     transpose_a=False,
                                     transpose_b=False)
        # Subtract resulting erase matrix from ones:
        erase_mat = tf.math.subtract(self.ones,
                                     erase_mat)

        # Multiply previous memory matrix with erase matrix (element-wise):
        new_memory = tf.math.multiply(self.memory,
                                      erase_mat)

        # Multiply weight vector with write vector:
        write_mat = tf.linalg.matmul(tf.reshape(weight_vec,
                                                (self.lgm_batch_size,
                                                 self.lgm_mem_locations,
                                                 1)),
                                     tf.reshape(write_vec,
                                                (self.lgm_batch_size,
                                                 1,
                                                 self.lgm_mem_size)),
                                     transpose_a=False,
                                     transpose_b=False)

        # Add write matrix to new memory matrix:
        new_memory = tf.math.add(new_memory,
                                 write_mat)

        # Apply dropout:
        new_memory = self.drop_memory(new_memory)
        weight_vec = self.drop_state_weights(weight_vec)
        erase_vec = self.drop_state_erase(erase_vec)
        write_vec = self.drop_state_write(write_vec)

        # Update memory and states:
        self.memory = new_memory
        self.state_weights = weight_vec
        self.state_erase = erase_vec
        self.state_write = write_vec

        # Concatenate output of first feedforward module and memory readout:
        mem_and_states = self.conc([read_vec,
                                    weight_vec,
                                    erase_vec,
                                    write_vec])

        # Activation of second feedforward module:
        f2 = self.drop2(mem_and_states)
        f2 = self.d2(f2)

        return self.reshape(f2)

    def erase_memory(self, batch_size=None, mem_locations=None, mem_size=None):
        """Re-initialise memory and recurrent states."""
        # Reset state of weights vector:
        vec_rand_01 = tf.random.normal((batch_size, mem_locations),
                                       mean=0.0,
                                       stddev=0.1,
                                       dtype=tf.float32)
        self.state_weights = vec_rand_01

        # Reset state of erase vector:
        vec_rand_02 = tf.random.normal((batch_size, mem_size),
                                       mean=0.0,
                                       stddev=0.1,
                                       dtype=tf.float32)
        self.state_erase = vec_rand_02

        # Reset state of write vector:
        vec_rand_03 = tf.random.normal((batch_size, mem_size),
                                       mean=0.0,
                                       stddev=0.1,
                                       dtype=tf.float32)
        self.state_write = vec_rand_03

        # Reset state of memory matrix:
        vec_rand_04 = tf.random.normal((batch_size, mem_locations, mem_size),
                                       mean=0.0,
                                       stddev=0.1,
                                       dtype=tf.float32)
        self.memory = vec_rand_04
