import pandas as pd
import numpy as np
import os
from os import path
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers


def build_ys():
    """
    This function loads the labels and builds a df 
    """
    y_df = pd.read_csv('../res/train.csv', header=0, dtype={'new_id':str, 'genre':np.int16})
    y_df = y_df.set_index('new_id')
    file_names = os.listdir('../res/train/')
    training_labels = np.zeros(len(file_names), dtype=np.int8)
 
    for i, file_name in enumerate(file_names):
        f_name, _ = path.splitext(file_name)
        file_label = y_df.loc[f_name].genre
        training_labels[i] = file_label
    np.save('../res/training_labels', training_labels)
    return training_labels

def load_x_y(data_file_name='../res/data_pca_ncom40.npy'):
    xs = np.load(data_file_name)
    ys = np.load('../res/training_labels.npy')
    print(ys.shape)
    print(xs.shape)
    #converting to categorical
    ys = keras.utils.to_categorical(ys)

    x_t, x_test, y_t, y_test = train_test_split(xs, ys, test_size=0.15, shuffle=True)
    return x_t, x_test, y_t, y_test

def build_model():
    model = keras.models.Sequential()
    model.add(layers.BatchNormalization(input_shape=(40,)))
    model.add(layers.Dense(75))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(100))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(50))
    model.add(layers.Dropout(0.2))
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(30))
    model.add(layers.Dropout(0.2))
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(6, activation='softmax'))
    optimizer_sgd = keras.optimizers.SGD(lr=0.003, decay=1e-6, momentum=0.9, nesterov=True)
    optimizer_adam = keras.optimizers.Adam()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer_sgd, metrics=['accuracy'])
    return model

def train_and_eval(model, x_train, y_train, x_eval, y_eval):
    history = model.fit(x_train, y_train, epochs=100, batch_size=128, validation_data=(x_eval, y_eval))
    score = model.evaluate(x_eval, y_eval)
    print(score)
    return history

model = build_model()
x_train, x_eval, y_train, y_eval = load_x_y()
print(y_train)
train_and_eval(model, x_train, y_train, x_eval, y_eval)