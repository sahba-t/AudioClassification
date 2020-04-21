import pandas as pd
import numpy as np
import os
from os import path
import numpy as np
import sklearn
import pickle
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

def load_x_y(data_file_name='../res/data_pca_ncom40.npy', test_split=0.15):
    xs = np.load(data_file_name)
    ys = np.load('../res/training_labels.npy')
    print(ys.shape)
    print(xs.shape)
    #converting to categorical
    ys = keras.utils.to_categorical(ys)

    if test_split > 0:
        x_t, x_test, y_t, y_test = train_test_split(xs, ys, test_size=test_split, shuffle=True)
        return x_t, x_test, y_t, y_test
    else:
        return xs, ys

def build_model():
    model = keras.models.Sequential()
    #model.add(layers.BatchNormalization(input_shape=(40,)))
    model.add(layers.Dense(60, input_dim=40))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(70))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.25))
    # model.add(layers.BatchNormalization())
    model.add(layers.Dense(50))
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(15))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(6, activation='softmax'))
    optimizer_sgd = keras.optimizers.SGD(lr=0.003, decay=1e-6, momentum=0.9, nesterov=True)
    optimizer_adam = keras.optimizers.Adam()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer_sgd, metrics=['accuracy'])
    return model

def train_and_eval(model, x_train, y_train, x_eval, y_eval):
    history = model.fit(x_train, y_train, epochs=1000, batch_size=128, validation_data=(x_eval, y_eval))
    score = model.evaluate(x_eval, y_eval)
    print(score)
    return history


def load_test_data(test_data_file_name='../res/test_pca_ncom40.npy', row_dict_name='../res/test_row_dict'):
    with open(row_dict_name, 'rb') as p_file:
        row_dict= pickle.load(p_file)

    eval_xs = np.load(test_data_file_name)
    return eval_xs, row_dict

def train_for_kaggle():
    model = build_model()
    x_train, y_train = load_x_y(test_split=0)
    model.fit(x_train, y_train, epochs=300, batch_size=128)
    x_test, row_dict = load_test_data()
    predictions = model.predict(x_test)
    with open('kaggle.csv', 'w') as csv_stream:
        csv_stream.write('id,genre\n')
        for r in range(predictions.shape[0]):
            predicted_genre = np.argmax(predictions[r,:])
            file_label = row_dict[r]
            csv_stream.write(f"{file_label},{predicted_genre}\n")

def validate_model():

    model = build_model()
    x_train, x_eval, y_train, y_eval = load_x_y()
    print(y_train)
    train_and_eval(model, x_train, y_train, x_eval, y_eval)

train_for_kaggle()
