import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, GlobalAvgPool2D
from tensorflow.keras.callbacks import EarlyStopping

(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

ss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
for train_ix, valid_ix in ss.split(X_train_full, y_train_full):
    X_train = X_train_full[train_ix]
    X_valid = X_train_full[valid_ix]
    y_train = y_train_full[train_ix]
    y_valid = y_train_full[valid_ix]

X_train = X_train.reshape(48000, 28, 28, 1)
X_valid = X_valid.reshape(12000, 28, 28, 1)

model = Sequential()

model.add(Conv2D(64, 7, activation="relu", padding="same", input_shape=[28,28,1]))
model.add(MaxPooling2D(2))
model.add(Conv2D(128, 3, activation="relu", padding="same"))
model.add(Conv2D(128, 3, activation="relu", padding="same"))
model.add(MaxPooling2D(2))
model.add(Conv2D(256, 3, activation="relu", padding="same"))
model.add(Conv2D(256, 3, activation="relu", padding="same"))
model.add(MaxPooling2D(2))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))

model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])

print(model.summary())

early_stop = EarlyStopping(patience=1, verbose=3, restore_best_weights=True)

model.fit(X_train, y_train, epochs=3, validation_data=(X_valid, y_valid), callbacks=[early_stop])



