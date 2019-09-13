#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Memory module for literature generating model using word embeddings."""


import numpy as np
import tensorflow as tf


# Shortcut for activation functions:
tanh = tf.keras.activations.tanh
sigmoid = tf.keras.activations.sigmoid
softmax = tf.keras.activations.softmax


class MeLa(tf.keras.layers.Layer):
    """Memory layer for literature generating model."""

    def __init__(self,  # noqa
                 batch_size,
                 input_size,
                 output_size,
                 mem_size,
                 drop_in,
                 drop_state,
                 drop_mem,
                 name='LiGeMo',
                 **kwargs):
        super(MeLa, self).__init__(name=name, **kwargs)

        # ** Memory input **

        # Dropout layers for state arrays:
        self.drop_mem = tf.keras.layers.Dropout(drop_mem)
        self.drop_state1 = tf.keras.layers.Dropout(drop_state)
        self.drop_state2 = tf.keras.layers.Dropout(drop_state)

        # Memory size is hardcoded (room for improvement):
        mem_size = 3 * input_size

        # Random values for initial state of memory input gate:
        vec_rand_01 = tf.random.normal((batch_size, mem_size),
                                       mean=0.0,
                                       stddev=0.5,
                                       dtype=tf.float32)

        # Memory input gate, recurrent input:
        self.mem_in_state = tf.Variable(initial_value=vec_rand_01,
                                        trainable=False,
                                        dtype=tf.float32,
                                        name='memory_input_state')

        # Dense layer controlling input to memory:
        self.dmi = tf.keras.layers.Dense(mem_size,
                                         activation=tanh,  # or use softmax?
                                         name='dense_memory_in')

        # ** Memory output **

        # Random values for initial state of memory output gate:
        vec_rand_02 = tf.random.normal((batch_size, output_size),
                                       mean=0.0,
                                       stddev=0.5,
                                       dtype=tf.float32)

        # Memory output gate, recurrent input:
        self.mem_out_state = tf.Variable(initial_value=vec_rand_02,
                                         trainable=False,
                                         dtype=tf.float32,
                                         name='memory_output_state')

        # Dense layer controlling memory output:
        self.dmo = tf.keras.layers.Dense(output_size,
                                         activation=tanh,
                                         name='dense_memory_out')

        # ** Memory **

        # Random values for initial state of memory vector:
        vec_rand_04 = tf.random.normal((batch_size, mem_size),
                                       mean=0.0,
                                       stddev=0.1,
                                       dtype=tf.float32)

        # The actual memory state:
        self.memory = tf.Variable(initial_value=vec_rand_04,
                                  trainable=False,
                                  dtype=tf.float32,
                                  name='memory_vector')

        # Math ops:
        self.add = tf.keras.layers.Add()
        self.sub = tf.keras.layers.Subtract()
        self.mult = tf.keras.layers.Multiply()
        self.conc = tf.keras.layers.Concatenate(axis=1)


    def call(self, inputs):  # noqa

        # Input dropout:
        inputs = self.drop1(inputs)

        # Recurrent layer controlling memory input:
        r1 = self.conc([self.mem_in_state,   # batch_size * mem_size
                        self.mem_out_state,  # batch_size * emb_size
                        inputs[:, 0, :],
                        self.memory])
        mem_in = self.dmi(r1)

        # Recurrent layer controlling memory output (read from memory):
        r2 = self.conc([self.mem_in_state,
                        self.mem_out_state,
                        inputs[:, 0, :],
                        self.memory])
        mem_out = self.dmo(r2)

        # Gated memory output:
        mem_out = self.mult([mem_out, self.memory])

        # Output of memory input layer gets added to memory:
        men_in_gated = self.mult([mem_in,
                                  self.conc([inputs[:, 0, :],
                                             inputs[:, 0, :],
                                             inputs[:, 0, :]
                                             ])
                                  ])
        new_memory = self.add([self.memory, men_in_gated])

        # Avoid excessive growth of memory vector:
        new_memory = tf.clip_by_value(new_memory,
                                      -100.0,
                                      100.0)

        # Update memory and states:
        self.memory = self.drop_mem(new_memory)
        self.mem_in_state = self.drop_state1(mem_in)
        self.mem_out_state = self.drop_state2(mem_out)

        # Only the reduced mean of the memory input gating is feed into
        # the second feedforward layer.
        # mem_in = tf.math.reduce_mean(mem_in,
        #                              axis=1,
        #                              keepdims=True)

        # Concatenate output of first feedforward module and updated memory
        # output:
        # f1_mem = self.conc([f1[:, 0, :], mem_out])  # no gradient
        # out = self.conc([f1[:, 0, :], mem_in, mem_out])

        return mem_out
