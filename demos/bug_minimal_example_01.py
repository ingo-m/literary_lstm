#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal example of bug using sample_weight with batch_size > 1."""

import numpy as np
import tensorflow as tf

batch_size = 32
sequence_len = 1
embedding_size = 100

x_train = np.random.randn(batch_size, sequence_len, embedding_size)
y_train = np.random.randn(batch_size, embedding_size)
sample_weight = np.random.randn(batch_size)

train_input = tf.keras.Input(shape=(sequence_len, embedding_size),
                             batch_size=batch_size)

lstm_layer = tf.keras.layers.LSTM(200,
                                  return_sequences=False,
                                  )(train_input)

dense_layer = tf.keras.layers.Dense(embedding_size,
                                    )(lstm_layer)

model = tf.keras.models.Model(inputs=train_input, outputs=dense_layer)

model.summary()

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
              loss=tf.keras.losses.mean_squared_error)

loss = model.train_on_batch(x_train,
                            y=y_train,
                            sample_weight=sample_weight)
