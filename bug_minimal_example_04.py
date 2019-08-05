#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test script for finding problem with non-converging LSTM in tf 1.14.0."""

import numpy as np
import tensorflow as tf

batch_size = 1

iterations = 100

# Some arbitrary sequence to learn:
train_seq = np.array([-1.0, 1.0, 0.0,
                      -0.5, 0.5, 0.0,
                      -2.0, 2.0, 0.0], dtype=np.float32)

sample_weight = None

train_input = tf.keras.Input(shape=(1, 1),
                             batch_size=batch_size)

lstm_layer = tf.keras.layers.LSTM(10,
                                  return_sequences=False,
                                  stateful=True,
                                  )(train_input)

dense_layer = tf.keras.layers.Dense(1,
                                    )(lstm_layer)

model = tf.keras.models.Model(inputs=train_input, outputs=dense_layer)

model.summary()

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
              loss=tf.keras.losses.mean_squared_error)

seq_len = train_seq.shape[0]
idx_train = 0

# Training
for idx_itr in range(iterations):

    loss = model.train_on_batch(train_seq[idx_train].reshape(1, 1, 1),
                                y=train_seq[(idx_train + 1)].reshape(1, 1),
                                sample_weight=sample_weight)

    # Increment / reset training index:
    if (idx_train + 2) < seq_len:
        idx_train += 1
    else:
        idx_train = 0

# Validation

# Validation model:
val_model = tf.keras.models.Model(inputs=train_input, outputs=dense_layer)

# Copy weights from training model to test model:
val_model.set_weights(model.get_weights())

new_sequence = []

x_pred = np.array(1.0, ndmin=3, dtype=np.float32)

for idx_new in range(30):
    y_pred = val_model.predict_on_batch(x_pred)
    x_pred[0, 0, 0] = y_pred[0, 0]
    new_sequence.append(str(np.around(y_pred[0, 0], decimals=2)))
print(' '.join(new_sequence))


