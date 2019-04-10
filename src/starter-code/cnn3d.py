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

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("white")

from sklearn.metrics import confusion_matrix, accuracy_score


class ClassifyWith3dCnn(object):
    def __init__(self, X_train, y_train, X_test, y_test):
        # Hyper Parameter
        self.batch_size = 86
        self.epochs = 30

        # Set up TensorBoard
        self.tensorboard = TensorBoard(batch_size=self.batch_size)

        self.X_train = self.translate(X_train).reshape(-1, 16, 16, 16, 3)
        self.X_test = self.translate(X_test).reshape(-1, 16, 16, 16, 3)

        self.y_train = to_categorical(y_train, 2)
        self.y_test = y_test

        self.model = self.CNN((16, 16, 16, 3), 2)

    def translate(self, x):
        xx = np.ndarray((x.shape[0], 4096, 3))
        for i in range(x.shape[0]):
            xx[i] = self.array_to_color(x[i])
            if i % 1000 == 0:
                print(i)
        # Free Memory
        del x

        return xx

    # Translate data to color
    def array_to_color(self, array, cmap="Oranges"):
        s_m = plt.cm.ScalarMappable(cmap=cmap)
        return s_m.to_rgba(array)[:, :-1]

    # Conv2D layer
    def Conv(
        self, filters=16, kernel_size=(3, 3, 3), activation="relu", input_shape=None
    ):
        if input_shape:
            return Conv3D(
                filters=filters,
                kernel_size=kernel_size,
                padding="Same",
                activation=activation,
                input_shape=input_shape,
            )
        else:
            return Conv3D(
                filters=filters,
                kernel_size=kernel_size,
                padding="Same",
                activation=activation,
            )

    # Define Model
    def CNN(self, input_dim, num_classes):
        model = Sequential()

        model.add(self.Conv(8, (3, 3, 3), input_shape=input_dim))
        model.add(self.Conv(16, (3, 3, 3)))
        # model.add(BatchNormalization())
        model.add(MaxPool3D())
        # model.add(Dropout(0.25))

        model.add(self.Conv(32, (3, 3, 3)))
        model.add(self.Conv(64, (3, 3, 3)))
        model.add(BatchNormalization())
        model.add(MaxPool3D())
        model.add(Dropout(0.25))

        model.add(Flatten())

        model.add(Dense(4096, activation="relu"))
        model.add(Dropout(0.5))

        model.add(Dense(1024, activation="relu"))
        model.add(Dropout(0.5))

        model.add(Dense(num_classes, activation="softmax"))

        return model

    # Train Model
    def train(self, optimizer, scheduler):

        print("Training...")
        self.model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        self.model.fit(
            x=self.X_train,
            y=self.y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.15,
            verbose=2,
            callbacks=[scheduler, self.tensorboard],
        )

    def evaluate(self):

        pred = self.model.predict(self.X_test)
        pred = np.argmax(pred, axis=1)

        print("Accuracy: ", accuracy_score(pred, self.y_test))
        # Heat Map
        array = confusion_matrix(self.y_test, pred)
        cm = pd.DataFrame(array, index=range(2), columns=range(2))
        plt.figure(figsize=(20, 20))
        sns.heatmap(cm, annot=True)
        plt.show()

    def cnn_initiate(self):

        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        scheduler = ReduceLROnPlateau(
            monitor="val_acc", patience=3, verbose=1, factor=0.5, min_lr=1e-5
        )

        self.train(optimizer, scheduler)
        self.evaluate()
