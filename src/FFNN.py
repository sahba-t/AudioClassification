import pandas as pd
import numpy as np
import os
from os import path
import numpy as np
import sklearn
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import librosa

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

def build_model(input_dim = 40):
    model = keras.models.Sequential()
    #model.add(layers.BatchNormalization(input_shape=(40,)))
    model.add(layers.Dense(80, input_dim=input_dim))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(100))
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
    history = model.fit(x_train, y_train, epochs=200, batch_size=128, validation_data=(x_eval, y_eval))
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
    
    x_train,x_eval, y_train,y_eval = load_x_y(test_split=0.1)
    model.fit(x_train, y_train, epochs=300, batch_size=128, validation_data=(x_eval, y_eval))
    x_test, row_dict = load_test_data()
    predictions = model.predict(x_test)
    with open('kaggle.csv', 'w') as csv_stream:
        csv_stream.write('id,genre\n')
        for r in range(predictions.shape[0]):
            predicted_genre = np.argmax(predictions[r,:])
            file_label = row_dict[r]
            csv_stream.write(f"{file_label},{predicted_genre}\n")




def extract_features_build_csv(folder_path="/users/sahba/scratch/git/project3/train", csv_file="../res/train_features.csv", train_mode=True):
    """
    Adapted from https://towardsdatascience.com/music-genre-classification-with-python-c714d032f0d8
    Use librosa to directly extract information from the wavefiles. Requires some music knowledge!
    """
    header = 'filename chroma_stft spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    if train_mode:
        header += ' label'
    header = header.replace(' ', ',')
    audio_files = list(os.listdir(folder_path))
    if train_mode:
        y_df = pd.read_csv('../res/train.csv', header=0, dtype={'new_id':str, 'genre':np.int16})
        y_df = y_df.set_index('new_id')
    else:
        row_dict = dict(zip(range(len(audio_files)), map(lambda x: path.splitext(x)[0], audio_files)))

    with open(csv_file, 'w') as csv_stream:
        csv_stream.write(header + "\n")
        for i, file_name in enumerate(audio_files):
            f_id, _ = path.splitext(file_name)
            if train_mode:
                file_genre = y_df.loc[f_id].genre
            y, sr = librosa.load(os.path.join(folder_path, file_name), mono=True, duration=30)
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            to_append = f'{file_name} {np.mean(chroma_stft)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
            for e in mfcc:
                to_append += f' {np.mean(e)}'
            if train_mode:
                to_append += f' {file_genre}'
            csv_stream.write(to_append.replace(" ", ",") + "\n")
            if i % 20 == 0:
                print(i)
    if not train_mode:
        with open('test_row_dict_features', 'wb')  as test_file:
            pickle.dump(row_dict, test_file)

def load_csv_train_model(csv_path):
    """"
    loads a csv of filename, extracted features, label and trains the model on it
    """
    data = pd.read_csv(csv_path)
    print(F"shape of data is {data.shape}")
    data = data.drop(['filename'],axis=1)
    scaler = StandardScaler()
    print('scaling data')
    X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))
    genre_list = data.iloc[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(genre_list)
    y =keras.utils.to_categorical(y)
    print('done with transforms')
    x_train, x_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2)
    model = build_model(input_dim=25)
    train_and_eval(model, x_train, y_train, x_eval, y_eval)
    return model

def feature_for_kaggle(train_csv="../res/train_features.csv"):
    model = load_csv_train_model(train_csv)
    test_data = pd.read_csv('../res/test_features.csv')
    file_names = test_data['filename']
    test_data = test_data.drop(['filename'], axis=1)
    scalar = StandardScaler()
    x_test = scalar.fit_transform(np.array(test_data, dtype=np.float))
    predictions = model.predict(x_test)
    with open('kaggle_feature.csv', 'w') as csv_stream:
        csv_stream.write('id,genre\n')
        for r in range(predictions.shape[0]):
            predicted_genre = np.argmax(predictions[r,:])
            file_label,_ = path.splitext(file_names.iloc[r])
            csv_stream.write(f"{file_label},{predicted_genre}\n")



def validate_model():
    model = build_model()
    x_train, x_eval, y_train, y_eval = load_x_y()
    print(y_train)
    train_and_eval(model, x_train, y_train, x_eval, y_eval)

# train_for_kaggle()
#extract_features_build_csv()
#extract_features_build_csv(folder_path="/users/sahba/scratch/git/project3/test", csv_file="../res/test_features.csv", train_mode=False)
# load_csv_train_model('../res/train_features.csv')
feature_for_kaggle()
