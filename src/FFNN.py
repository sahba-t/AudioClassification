import pandas as pd
import numpy as np
import os
from os import path
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import librosa
import math

def plot_conf_matrix(array_file="../res/confMat/conf_matrix.npy", conf_array=None, out_file_name="cm.png"):
    """
    given a confusion matrix as a numpy array (either saved on disk or actual array)
    plot the confusion matrix as a heatmap
    saves the heatmap in the res folder
    """
    import matplotlib
    import matplotlib.pyplot as plt

    if not conf_array:
        conf_array = np.load(array_file)

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
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, "%.2f" % conf_array[i, j],ha="center", va="center", color="w")

    ax.set_title("The Confusion Matrix")
    fig.tight_layout()
    plt.savefig("../res/confMat/" + out_file_name, bbox_inches='tight', pad_inches=0.3)
    plt.show()





def build_ys():
    """
    This function loads the labels for the training data and 
    returns the classes of each training datapoint
    used for classification witht the PCA data
    @returns a numpy array where the value at index `i` corresponds to the 
    genre of training sample i  
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

def load_x_y(data_file_name, test_split=0.20, convert_2_cat=True):
    """
    loads the training data that has been reduced by pca 
    returns the training data and labels
    """
    xs = np.load(data_file_name)
    ys = np.load('../res/training_labels.npy')
    print(ys.shape)
    print(xs.shape)
    #converting to categorical
    if convert_2_cat:
        ys = keras.utils.to_categorical(ys)

    if test_split > 0:
        x_t, x_test, y_t, y_test = train_test_split(xs, ys, test_size=test_split, shuffle=True)
        return x_t, x_test, y_t, y_test
    else:
        return xs, ys

def build_model(input_dim=40, loss='categorical_crossentropy'):
    model = keras.models.Sequential()
    model.add(layers.BatchNormalization(input_shape=(input_dim,)))
    model.add(layers.Dense(60, input_dim=input_dim))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.15))
    #model.add(layers.BatchNormalization())

    #layer2
    model.add(layers.Dense(100))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))
    #model.add(layers.BatchNormalization())

    #layer 3
    model.add(layers.Dense(50))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.15))
    #model.add(layers.BatchNormalization())
    
    #layer 4
    #model.add(layers.Dense(15, activation='relu'))
    model.add(layers.Dense(15))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())

    #output layer
    model.add(layers.Dense(6, activation='softmax'))
    optimizer_sgd = keras.optimizers.SGD(lr=0.003, decay=1e-6, momentum=0.9, nesterov=True)
    optimizer_adam = keras.optimizers.Adam(learning_rate=0.003)
    model.compile(loss=loss, optimizer=optimizer_sgd, metrics=['accuracy'])
    return model

def nn_with_pca(pca_array='../res/pca/data_pca_40.npy', num_comp=40, do_conf_mat=False, do_kfold=0):       
    if do_kfold > 1:
        from sklearn.model_selection import StratifiedKFold
        seed = 17
        kfold = StratifiedKFold(n_splits=do_kfold, shuffle=True, random_state=seed)
        csv_scores = np.zeros(do_kfold)
        X = np.load(pca_array)
        y = np.load('../res/training_labels.npy')
        print(f'started {do_kfold}-folds validation. keeping {len(X)} for evaluation')
        for i,(train, test) in enumerate(kfold.split(X, y)):
            y_train = keras.utils.to_categorical(y[train])
            y_test = keras.utils.to_categorical(y[test])
            model = build_model(input_dim=num_comp)
            model.fit(X[train], y_train, epochs=100, batch_size=64, verbose=False)
            result = model.evaluate(X[test], y_test, verbose=False)
            score = result[-1] * 100
            print("%s: %.2f%%" % (model.metrics_names[1], score))
            csv_scores[i] = score
        print("average: %.2f%% +- %.2f%%" % (csv_scores.mean(), csv_scores.std()))

    x_train, x_eval, y_train, y_eval = load_x_y(pca_array, 0.2)
    model = build_model(input_dim=num_comp)
    "given the training and evbaluation data, trains the model"  
    history = model.fit(x_train, y_train, epochs=100, batch_size=64)
    score = model.evaluate(x_eval, y_eval)[-1]
    calc_CI(score, x_eval.shape[0])
    cat_2_num = lambda x: [np.argmax(r) for r in x]
    if do_conf_mat:
        build_conf_matrix(model, x_eval, y_eval, F'../res/pca_cm_{str(num_comp)}', cat_2_num, cat_2_num)


    return history


def load_test_data_pca(test_data_file_name='../res/test_pca_ncom40.npy', row_dict_name='../res/test_row_dict'):
    with open(row_dict_name, 'rb') as p_file:
        row_dict= pickle.load(p_file)

    eval_xs = np.load(test_data_file_name)
    return eval_xs, row_dict

def train_for_kaggle_pca():
    model = build_model()
    
    x_train,x_eval, y_train,y_eval = load_x_y('../res/pca/data_pca_40.npy', test_split=0.1)
    model.fit(x_train, y_train, epochs=300, batch_size=64, validation_data=(x_eval, y_eval))
    x_test, row_dict = load_test_data_pca()
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
    if applying to training data set train_mode=False
    @param folder_path(str): the path of the training wav files
    @param csv_file(str): the path of the csv file to output the features
    @paream train_mode(boolean): if True, the labels will be appended to the csv file 
    """
    header = 'filename chroma_stft spectral_centroid spectral_bandwidth rolloff zero_crossing_rate rms'
    for i in range(1, 21):
        header += f' mfcc{i}'
    if train_mode:
        header += ' label'
    header = header.replace(' ', ',')
    audio_files = list(os.listdir(folder_path))
    if train_mode:
        y_df = pd.read_csv('../res/train.csv', header=0, dtype={'new_id':str, 'genre':np.int16})
        y_df = y_df.set_index('new_id')
    
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
            rms = librosa.feature.rms(y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            to_append = f'{file_name} {np.mean(chroma_stft)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)} {np.mean(rms)}'    
            for e in mfcc:
                to_append += f' {np.mean(e)}'
            if train_mode:
                to_append += f' {file_genre}'
            csv_stream.write(to_append.replace(" ", ",") + "\n")
            if i % 20 == 0:
                print(i)


def load_csv_features(csv_path='../res/train_features.csv', cat_2_num=True, shuffle=True):
    """
    reads the csv, standardize the data and converts genres to ints
    returns the standardized features and y as ints representing categories
    """
    data = pd.read_csv(csv_path)
    data = data.sample(frac=1.0)
    print(F"shape of data is {data.shape}")
    data = data.drop(['filename'], axis=1)
    scaler = StandardScaler()
    print('scaling data')
    
    #standardization: we do this so that we can use the same transform for the test set
    train_data_raw = np.array(data.iloc[:, :-1], dtype=float)
    scaler = scaler.fit(train_data_raw)
    X = scaler.transform(train_data_raw)
    genre_list = data.iloc[:, -1]
    if cat_2_num:
        encoder = LabelEncoder()
        y = encoder.fit_transform(genre_list)
    else:
        y = genre_list
    print('done with transforms')
    return X, y, scaler
    

def nn_cross_val(csv_path='../res/train_features.csv', input_dim = 26, folds=5, save_models=False):
    """
    cross validation for the neural network based on features
    adapted from:
    https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/
    """
    from sklearn.model_selection import StratifiedKFold
    seed = 17
    X, y, scaler = load_csv_features(csv_path)
    kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    csv_scores = np.zeros(folds)
    print(f'started {folds}-folds training')
    print(y.shape)
    models = None 
    if save_models:
        models = [None] * folds

    for i,(train, test) in enumerate(kfold.split(X, y)):
        y_train = keras.utils.to_categorical(y[train])
        y_test = keras.utils.to_categorical(y[test])
        model = build_model(input_dim=input_dim)
        model.fit(X[train], y_train, epochs=200, batch_size=64, verbose=False)
        result = model.evaluate(X[test], y_test, verbose=False)
        score = result[-1] * 100
        print("%s: %.2f%%" % (model.metrics_names[1], score))
        csv_scores[i] = score
        if save_models:
            models[i] = model
    print("average: %.2f%% +- %.2f%%" % (csv_scores.mean(), csv_scores.std()))
    return models, scaler

def load_csv_train_NN(csv_path, input_dim=40, do_conf_mat=False, plot_training=False):
    """"
    loads a csv of filename, extracted features, label and trains the model on it
    Will print the confusion matrix if do_conf_mat is True
    returns the trained model along with the scaler used to standardize the data
    """
    X, y, scaler = load_csv_features(csv_path)
    x_train, x_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2)

    print(F"\nusing {x_train.shape[0]} training samples\n")

    y_onehot_train = keras.utils.to_categorical(y_train)
    y_onehot_eval = keras.utils.to_categorical(y_eval)
    
    #building the model and training
    model = build_model(input_dim=input_dim)
    history = model.fit(x_train, y_onehot_train, epochs=200, batch_size=64, validation_data=(x_eval, y_onehot_eval))
    result = model.evaluate(x_eval, y_onehot_eval)
    calc_CI(result[-1], x_eval.shape[0])
    #prepare the confusion matrix if requested
    if do_conf_mat:
        from sklearn.metrics import confusion_matrix
        print('preparing the conf matrix')
        predictions = model.predict(x_eval)
        y_pred = [np.argmax(predictions[r]) for r in range(predictions.shape[0])]
        # print a conf matrix and normalize each row!
        cmx = confusion_matrix(y_eval, y_pred, normalize='true')
        print(cmx)
        np.save('../res/conf_matrix', cmx)
    if plot_training:
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.rcParams.update({'font.size': 14})
        plt.plot(history.history['accuracy'], linewidth=3)
        plt.plot(history.history['val_accuracy'], linewidth=3)
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['training', 'evaluation'], loc='upper left')
        plt.savefig("../res/train_acc.png",  bbox_inches='tight', pad_inches=0.3)
        plt.figure()
        # summarize history for loss
        plt.plot(history.history['loss'], linewidth=3)
        plt.plot(history.history['val_loss'], linewidth=3)
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['training', 'evaluation'], loc='upper left')
        plt.savefig("../res/train_loss.png",  bbox_inches='tight', pad_inches=0.3)
        plt.show()
    return model, scaler


def build_conf_matrix(model, x_eval, y_eval, output_file, fn_on_preds=None, fn_on_yeval=None):
    """
    given a model and and evaluation set generates the confusion matrix
    if needed a function can be passed to be applied to the result of prediction
    (such as interperating the result of one-hot-enocded results)
    saveds the confusion matrix to disk and also returns the matrix
    """
    from sklearn.metrics import confusion_matrix
    print('preparing the conf matrix')
    predictions = model.predict(x_eval)
    if fn_on_preds:
        y_pred = fn_on_preds(predictions)
    else:
        y_pred = predictions
    if fn_on_yeval:
        y_eval = fn_on_yeval(y_eval)
    # print a conf matrix and normalize each row!
    cmx = confusion_matrix(y_eval, y_pred, normalize='true')
    print(cmx)
    print('saving cmx to' + str(output_file))
    np.save(output_file, cmx)
    return cmx

def predict_kaggle_feature(train_csv="../res/train_features.csv", test_csv='../res/test_features.csv', output_csv='kaggle_feature.csv'):
    """
    Trains the model based on the features extracted from the training files
    and uses them to classify the test set
    writes the prediction to ./kaggle_features.csv by default
    """
    #using the same scaler to transform the test set
    model, scaler = load_csv_train_NN(train_csv, input_dim=26)
    test_data = pd.read_csv(test_csv)
    file_names = test_data['filename']
    test_data = test_data.drop(['filename'], axis=1)
    x_test = scaler.transform(np.array(test_data, dtype=np.float))
    predictions = model.predict(x_test)
    print("!!!!! ATTENTION THE SHAPE IS:")
    print(predictions.shape)
    with open(output_csv, 'w') as csv_stream:
        csv_stream.write('id,genre\n')
        for r in range(predictions.shape[0]):
            predicted_genre = np.argmax(predictions[r, :])
            file_label, _ = path.splitext(file_names.iloc[r])
            csv_stream.write(f"{file_label},{predicted_genre}\n")

def ensemble_for_kaggle(output_csv='kaggle_ensemble.csv'):
    from sklearn import tree
    dt_count = 5
    dt_models = []
    models, scaler = nn_cross_val(save_models=True)
    for i in range(dt_count):
        X, y,_ = load_csv_features(cat_2_num=False)
        x_train, x_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2)
        clf = tree.DecisionTreeClassifier(max_depth=8)
        clf = clf.fit(x_train, y_train)
        dt_models.append(clf)
    test_data = pd.read_csv('../res/test_features.csv')
    file_names = test_data['filename']
    test_data = test_data.drop(['filename'], axis=1)
    x_test = scaler.transform(np.array(test_data, dtype=np.float))
    predictions = [model.predict(x_test) for model in models]
    dt_predictions = [model.predict(x_test) for model in dt_models]
    number_of_test_cases = predictions[0].shape[0]
    with open(output_csv, 'w') as csv_stream:
        csv_stream.write('id,genre\n')
        for test_case in range(number_of_test_cases):
            votes = np.zeros(6, dtype=np.int16)
            for nn_prediction in predictions:
                    predicted_genre = np.argmax(nn_prediction[test_case, :])
                    votes[predicted_genre] += 1
            for dt_prediction in dt_predictions:
                votes[dt_prediction[test_case]] += 1
            voted_genre = np.argmax(votes)
            print("genre %d selected with %d votes" % (voted_genre, votes[voted_genre]))
            file_label, _ = path.splitext(file_names.iloc[test_case])
            csv_stream.write(f"{file_label},{voted_genre}\n")
    

def DTree_Implementation(do_conf_mat=False, max_depth=None, data_type='FEATURES',
                        pca_array='../res/pca/data_pca_40.npy', plot_cm=False):
    """
    Performs decision tree classifier with either pca data or domain specific data
    can generate confusion matrix
    """
    
    
    from sklearn import tree
    if data_type.upper() == 'FEATURES':
        X, y, _ = load_csv_features(cat_2_num=False)
        x_train, x_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2)
    elif data_type.upper() == 'PCA':
        x_train, x_eval, y_train, y_eval = load_x_y(pca_array, test_split=0.20, convert_2_cat=False)
        clf = clf.fit(x_train, y_train)
    else:
        print("ERROR: unsupported data representation " + data_type)
    clf = tree.DecisionTreeClassifier(max_depth=max_depth)
    eval_n = x_eval.shape[0]
    print(F"training the decision tree on {x_train.shape[0]} instances")
    clf = clf.fit(x_train, y_train)
    print(F"Decision Tree fit is complete, the depth is {clf.get_depth()}"
          + "and the score on training set is:")
    print(clf.score(x_train, y_train))
    print(F"and on the eval set {eval_n} instances:")
    clf_score = clf.score(x_eval, y_eval)
    calc_CI(clf_score, x_eval.shape[0])
    if do_conf_mat:
        array_dst = F'../res/confMat/dt_cm_{str(max_depth)}'
        build_conf_matrix(clf, x_eval, y_eval, array_dst)
        if plot_cm:
            plot_conf_matrix(array_file=array_dst + ".npy", conf_array=None, out_file_name=F"dt_cm_{str(max_depth)}" + ".png")
            
    return clf_score
    

def calc_CI(score, sample_size):
    """
    Calculate and prints the confidence interval given the score and the size of the evaluation set
    resulted in that accuracy
    @param score(float)
    @param sample_size (int)
    @return the confidence interval
    """
    CI = math.sqrt(score * (1 - score)/sample_size) * 100
    print('Score: %.2f +- CI = %.2f' % (score, CI))
    return CI


# train_for_kaggle_pca()
#building the csv for training data
#extract_features_build_csv()
#building the csv for the test data
#extract_features_build_csv(folder_path="/users/sahba/scratch/git/project3/test", csv_file="../res/test_features.csv", train_mode=False)
#load_csv_train_model('../res/train_features.csv', do_conf_mat=True, input_dim=26)
#plot_conf_matrix()

# to test the neural netwok performance on extracted features
load_csv_train_NN('../res/train_features.csv', do_conf_mat=False, input_dim=26, plot_training=True)

#cross validation neural network features
# nn_cross_val('../res/train_features.csv')

#use the NN to output predictions to kaggle
#predict_kaggle_feature()

#trying the decision tree
# DTree_Implementation(max_depth=10, do_conf_mat=True, plot_cm=True, data_type="PCA")
# nn_with_pca(do_conf_mat=True, do_kfold=5)

#plotting confusion matrix
#plot_conf_matrix(array_file='../res/confMat/pca_cm_40.npy', out_file_name="pca_40_cm.png")
# ensemble_for_kaggle()