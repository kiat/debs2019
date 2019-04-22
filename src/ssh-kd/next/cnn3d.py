from __future__ import division, print_function, absolute_import

from keras.models import Sequential, model_from_json
from keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Conv3D,
    MaxPool3D,
    BatchNormalization,
    Input,
)
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from keras.models import load_model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.metrics import confusion_matrix, accuracy_score


class ClassifyWith3dCnn(object):
    def __init__(self, num_classes):
        # Hyper Parameter
        self.batch_size = 86
        self.epochs = 86
        self.num_classes = num_classes

        # Set up TensorBoard
        self.tensorboard = TensorBoard(batch_size=self.batch_size)
        self.model = self.CNN((7, 10, 10, 3), self.num_classes)

    def set_inputs(self, X_train, y_train, X_test, y_test, num_classes):
        self.X_train = self.translate(X_train)
        self.X_train = self.X_train.reshape(X_train.shape[0], 7, 10, 10, 3)
        self.X_test = self.translate(X_test)
        self.X_test = self.X_test.reshape(X_test.shape[0], 7, 10, 10, 3)

        self.y_train = y_train
        self.y_test = y_test

    def translate(self, x):
        translate_ = np.ndarray((x.shape[0], 7000, 3))
        for i in range(x.shape[0]):
            translate_[i] = self.array_to_color(x[i])
        del x
        return translate_

    # Translate data to color
    def array_to_color(self, array, cmap="Greys"):
        s_m = plt.cm.ScalarMappable(cmap=cmap)
        return s_m.to_rgba(array)[:, :-1]

    # Define Model
    def CNN(self, input_dim, num_classes):
        model = Sequential()

        model.add(Conv3D(filters=8, kernel_size=(3, 3, 3), activation="relu"))
        model.add(Conv3D(filters=16, kernel_size=(3, 3, 3), activation="relu"))
        # model.add(BatchNormalization())
        model.add(MaxPool3D(pool_size=(2,2,2)))
        # model.add(Dropout(0.25))

        model.add(Conv3D(filters=32, kernel_size=(3, 3, 3), activation="relu"))
        model.add(Conv3D(filters=64, kernel_size=(3, 3, 3), activation="relu"))
        # model.add(BatchNormalization())
        model.add(MaxPool3D())
        # model.add(Dropout(0.25))

        model.add(BatchNormalization())
        model.add(Flatten())

        model.add(Dense(2048, activation="relu"))
        model.add(Dropout(0.5))

        model.add(Dense(512, activation="relu"))
        model.add(Dropout(0.5))

        model.add(Dense(num_classes, activation="softmax"))

        return model

    # Train Model
    def train(self, optimizer, scheduler):
        print("Training...")
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        self.model.fit(
            x=self.X_train,
            y=self.y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.2,
            callbacks=[scheduler, self.tensorboard],
        )

    def evaluate(self, X_test):
        X_test = self.translate(X_test)
        X_test = X_test.reshape(X_test.shape[0], 16, 16, 16, 3)

        pred = self.model.predict(X_test)
        pred = np.argmax(pred, axis=1)

        # print("Accuracy: ", accuracy_score(pred, y_test))
        return pred

    def cnn_initiate(self):
        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        scheduler = ReduceLROnPlateau(monitor="val_acc", patience=3, verbose=1, factor=0.5, min_lr=1e-5)

        self.train(optimizer, scheduler)

    def save_model(self):
        self.model.save('../model/3dcnn.h5')

    def upload_model(self,path_to_model):
        self.model = load_model(path_to_model)