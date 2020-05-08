import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from skimage import io
from random import shuffle

train_spectro_path = '../res/spectrogram/train/'
test_spectro_path = '../res/spectrogram/test/'

train_spectro_names = os.listdir(train_spectro_path)
test_spectro_names = os.listdir(test_spectro_path)

expected_spectro_shape = (128, 2551)
num_classes = 6

img_rows = expected_spectro_shape[0]
img_cols = expected_spectro_shape[1]

print("num train spectros:", len(train_spectro_names))
print("num test spectros:", len(test_spectro_names))
print("expected_spectro_shape:", expected_spectro_shape)


def plot_conf_matrix(conf_array=None):
    fig, ax = plt.subplots()
    im = ax.imshow(conf_array)
    labels = "Rock,Pop,Folk,Instr,Elec,HH".split(',')
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, "%.2f" % conf_array[i, j], ha="center", va="center", color="w")

    ax.set_title("The Confusion Matrix")
    fig.tight_layout()
    plt.savefig("../res/confMat_cnn.png", bbox_inches='tight', pad_inches=0.3)
    plt.show()


def read_labels(f_names: list):
    labels = np.zeros(len(f_names))
    y_df = pd.read_csv('../res/train.csv', header=0, dtype={'new_id': str, 'genre': np.int16})
    y_df = y_df.set_index('new_id')
    for i in range(len(f_names)):
        labels[i] = y_df.loc[f_names[i][:-4]].genre
    print("Finished reading", len(labels), 'labels')
    return labels


def read_spectrogram(path: str, f_names: list):
    img_data = np.zeros(shape=(len(f_names), expected_spectro_shape[0], expected_spectro_shape[1]))
    for i in range(len(f_names)):
        spectro = io.imread(path + f_names[i][:-3] + 'png')

        if spectro.shape[1] > expected_spectro_shape[1]:
            spectro = spectro[:, :(expected_spectro_shape[1] - spectro.shape[1])]
        elif spectro.shape[1] < expected_spectro_shape[1]:
            padding_matrix_shape = (expected_spectro_shape[0], expected_spectro_shape[1] - spectro.shape[1])
            spectro = np.hstack((spectro, np.zeros(padding_matrix_shape)))

        img_data[i] = spectro
        if expected_spectro_shape != img_data[i].shape:
            print("index:", i, "has shape", img_data[i].shape)
    print("Spectrogram from", path, "read in! Shape is:", img_data.shape)
    return img_data


def format_data(data):
    img_rows = expected_spectro_shape[0]
    img_cols = expected_spectro_shape[1]
    if K.image_data_format() == 'channels_first':
        data_x = data.reshape(data.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        data_x = data.reshape(data.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    return data_x, input_shape


def create_1D_CNN(input_shape):
    model = keras.models.Sequential()
    model.add(layers.Conv1D(filters=8, kernel_size=16, activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool1D(2))

    model.add(layers.Conv1D(filters=8, kernel_size=16, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool1D(2))
    model.add(layers.Dropout(0.2))

    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=.09),
                  metrics=['accuracy'])

    return model


def calc_CI(score, sample_size):
    print('CI = %.2f' % (math.sqrt(score * (1 - score) / sample_size) * 100))


if __name__ == '__main__':
    def main():
        training_x = read_spectrogram(train_spectro_path, train_spectro_names)
        training_labels = read_labels(train_spectro_names)

        shuffle_training_x_labs = list(zip(training_x, training_labels))
        shuffle(shuffle_training_x_labs)

        training_x = np.array([x for x, _ in shuffle_training_x_labs])
        training_labels = np.array([l for _, l in shuffle_training_x_labs])

        train_size = int(len(training_x) * .85)

        train_set_x = training_x[:train_size]
        train_set_y = training_labels[:train_size]
        eval_set_x = training_x[train_size:]
        eval_set_y = training_labels[train_size:]
        print("\n\n")
        print(training_x.shape)
        print(train_set_y.shape)
        print(eval_set_y.shape)
        print("\n\n")

        # model = create_CNN_1d(input_shape=(img_rows, img_cols)) # Sahba's
        model = create_1D_CNN(input_shape=(img_rows, img_cols))  # Mauricio's
        model.fit(train_set_x, train_set_y,
                  epochs=50,
                  # batch_size=50,
                  shuffle=True,  # @TODO make true when you fine a high acc on current architecture
                  verbose=1,
                  validation_data=(eval_set_x, eval_set_y),
                  use_multiprocessing=True)

        model.summary()
        score = model.evaluate(training_x[train_size:], eval_set_y, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        if score[1] > .6:
            testing_x = read_spectrogram(test_spectro_path, test_spectro_names)
            predictions = model.predict(testing_x)
            with open('../results/CNN_' + str(score[1]) + '.csv', 'w') as csv_stream:
                csv_stream.write('id,genre\n')
                for r in range(predictions.shape[0]):
                    predicted_genre = np.argmax(predictions[r, :])
                    file_label = test_spectro_names[r][:-4]
                    csv_stream.write(f"{file_label},{predicted_genre}\n")
            print('File written to:', '../results/CNN_' + str(score[1]) + '.csv')


    main()
