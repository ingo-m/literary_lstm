#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test script for finding problem with non-converging LSTM in tf 1.14.0."""

import numpy as np
import tensorflow as tf

batch_size = 1

iterations = 100000

# Some arbitrary sequence to learn:
train_seq = np.array([1.0, 1.0, 1.0,
                      0.0, 0.0, 0.0], dtype=np.float32)

sample_weight = np.ones(1, dtype=np.float32)

train_input = tf.keras.Input(shape=(1, 1),
                             batch_size=batch_size)

lstm_01 = tf.keras.layers.LSTM(10,
                               return_sequences=True,
                               stateful=True,
                               )(train_input)

lstm_02 = tf.keras.layers.LSTM(10,
                               return_sequences=False,
                               stateful=True,
                               )(lstm_01)

dense_layer = tf.keras.layers.Dense(1,
                                    )(lstm_02)

model = tf.keras.models.Model(inputs=train_input, outputs=dense_layer)

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
              loss=tf.keras.losses.MeanSquaredError())
              #loss=tf.keras.losses.mean_squared_error)
              #loss=tf.losses.mean_squared_error)

# Validation model:
val_model = tf.keras.models.Model(inputs=train_input, outputs=dense_layer)

seq_len = train_seq.shape[0]
idx_x = 0
idx_y = 1

# Training
for idx_itr in range(iterations):

    loss = model.train_on_batch(train_seq[idx_x].reshape(1, 1, 1),
                                y=train_seq[idx_y].reshape(1, 1),
                                sample_weight=sample_weight)

    # Increment / reset training index:
    idx_x += 1
    idx_y += 1
    if seq_len <= idx_x:
        idx_x = 0
    if seq_len <= idx_y:
        idx_y = 0

    # Validation
    if (idx_itr % 10000 == 0):

        # Copy weights from training model to test model:
        val_model.set_weights(model.get_weights())

        new_sequence = []

        x_pred = train_seq[idx_x].reshape(1, 1, 1)

        for idx_new in range(20):
            y_pred = val_model.predict_on_batch(x_pred)
            x_pred[0, 0, 0] = y_pred[0, 0]
            new_sequence.append(str(np.around(y_pred[0, 0], decimals=2)))
        print(' '.join(new_sequence))

        model.reset_states()
        val_model.reset_states()
