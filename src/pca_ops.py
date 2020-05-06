import scipy
import scipy.io.wavfile as wavfile
import sklearn
from sklearn.decomposition import PCA
import pickle
import os
import numpy as np
import datetime 
def load_data(save=False):
    train_dir = '/users/sahba/scratch/git/project3/train/'
    file_names = os.listdir(train_dir)
    training_data = np.zeros((len(file_names) ,1439471))
    #for i,file in enumerate(files_names):
    for i, file in enumerate(file_names):
        _, data = wavfile.read(train_dir + file)
        if i % 100 == 0:
            print(i)
            
        if len(data.shape) > 1:
            training_data[i,:data.shape[0]] = data[:,0]
        else:
            training_data[i, :data.shape[0]] = data

    print('done loading')
    currentDT = datetime.datetime.now()
    if save:
        np.save('./data_padded' + str(currentDT), training_data)
    return training_data

def do_pca(training_data, kwargs): 

    pca = PCA(**kwargs)
    pca.fit(training_data)
    print(pca)
    with open("./pca_pickled_80", 'wb') as pca_file:
        pickle.dump(pca, pca_file)


def load_pca(filename):
    with open(filename, 'rb') as p_file:
        pca= pickle.load(p_file)
    return pca


def transform_data(pca, data, save=True, file_name='pcad_data'):
    print('applying pca on data')
    data_pca = pca.transform(data)
    print(type(data_pca))
    print(data_pca.shape)
    if save:
        currentDT = datetime.datetime.now()
        np.save(file_name + str(currentDT), data_pca)
        


if __name__ == '__main__':
    training_data = load_data()
    # print('now pca')
    training_data = load_data()
    do_pca(training_data, {'n_components':80})
    #pca = load_pca('pca_pickled')
    #transform_data(pca, training_data, file_name='data_PCA10')
