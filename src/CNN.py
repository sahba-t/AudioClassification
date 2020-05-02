import os
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from skimage import io

train_wav_path = '../res/wav/train/'
test_wav_path = '../res/wav/test/'
train_spectro_path = '../res/spectrogram/train/'
test_spectro_path = '../res/spectrogram/test/'

train_wav_names = os.listdir(train_wav_path)
test_wav_names = os.listdir(test_wav_path)
train_spectro_names = os.listdir(train_spectro_path)
test_spectro_names = os.listdir(test_spectro_path)

expected_spectro_shape = (128, 2551)
num_classes = 6

print("num train wavs:", len(train_wav_names))
print("num test wavs:", len(test_wav_names))
print("num train spectros:", len(train_spectro_names))
print("num test spectros:", len(test_spectro_names))
print("expected_spectro_shape:", expected_spectro_shape)


def read_labels(f_names: list):
    labels = np.zeros(len(f_names))
    y_df = pd.read_csv('../res/train.csv', header=0, dtype={'new_id':str, 'genre':np.int16})
    y_df = y_df.set_index('new_id')
    for i in range(len(f_names)):
        labels[i] = y_df.loc[f_names[i][:-4]].genre
    print("Finished reading", len(labels), 'labels')
    return labels


def read_spectrogram(path: str, f_names: list):
    img_data = np.zeros(shape=(len(f_names), expected_spectro_shape[0], expected_spectro_shape[1]))
    for i in range(len(f_names)):
        img_data[i] = io.imread(path + f_names[i][:-3] + 'png')
        if expected_spectro_shape != img_data[i].shape:
            print("index:", i, "has shape", img_data[i].shape)
    print("Spectrogram from", path, "read in! Shape is:", img_data.shape)
    return img_data


def create_CNN(input_shape=None):
    model = keras.models.Sequential()
    model.add(layers.Conv2D(filters=8,
                            kernel_size=(128, 4),
                            activation='relu',
                            padding='same',
                            input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=3, strides=1))
    model.add(layers.Dropout(0.2))

    # model.add(layers.Conv2D(filters=4,
    #                         kernel_size=8,
    #                         activation='relu'))
    # model.add(layers.BatchNormalization())
    # model.add(layers.MaxPool2D(pool_size=3, strides=1))
    # model.add(layers.Dropout(0.15))

    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    def main():
        training_x = read_spectrogram(train_spectro_path, train_spectro_names)
        training_labels = read_labels(train_wav_names)
        # testing_x = read_spectrogram(test_spectro_path, test_spectro_names)

        train_size = int(len(training_x) * .8)
        train_set_x = training_x[:train_size]
        train_set_y = training_labels[:train_size]

        eval_set_x = training_x[train_size:]
        eval_set_y = training_labels[train_size:]

        img_rows = expected_spectro_shape[0]
        img_cols = expected_spectro_shape[1]
        if K.image_data_format() == 'channels_first':
            train_set_x = train_set_x.reshape(train_set_x.shape[0], 1, img_rows, img_cols)
            eval_set_x = eval_set_x.reshape(eval_set_x.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            train_set_x = train_set_x.reshape(train_set_x.shape[0], img_rows, img_cols, 1)
            eval_set_x = eval_set_x.reshape(eval_set_x.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

        model = create_CNN(input_shape=input_shape)
        model.fit(train_set_x, train_set_y,
                  epochs=7,
                  batch_size=50,
                  verbose=1,
                  validation_data=(eval_set_x, eval_set_y),
                  use_multiprocessing=True)
        score = model.evaluate(eval_set_x, eval_set_y, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
    main()
