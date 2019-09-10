
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for demonstrating that cell states of stateful LSTM are not
copied when copying model weights.
"""

import numpy as np
import tensorflow as tf

# First model:
y_1 = tf.keras.Input(shape=(1, 1),
                     batch_size=1)
#x_1 = tf.keras.layers.Dense(1)(y_1)
x_1 = tf.keras.layers.LSTM(1, stateful=True)(y_1)
model_1 = tf.keras.models.Model(inputs=y_1, outputs=x_1)
model_1.summary()

# Identical second model:
y_2 = tf.keras.Input(shape=(1, 1),
                     batch_size=1)
#x_2 = tf.keras.layers.Dense(1)(y_2)
x_2 = tf.keras.layers.LSTM(1, stateful=True)(y_2)
model_2 = tf.keras.models.Model(inputs=y_2, outputs=x_2)
model_2.summary()

x = np.ones((1, 1, 1))

model_1.predict_on_batch(x)

model_2.predict_on_batch(x)

model_2.set_weights(model_1.get_weights())

model_1.predict_on_batch(x)

model_2.predict_on_batch(x)
