import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def ex_feature_description():
    return pd.read_csv('feature_description.csv', sep=',').values

def ex1_data(train_X, train_y, test_X, test_y):
    X_train = np.array(pd.read_csv('train/' + train_X[0], sep=','))
    y_train = np.array(pd.read_csv('train/' + train_y[0], sep=',')) 

    for i in range(1, len(train_X)):
        X_temp = np.array(pd.read_csv('train/' + train_X[i], sep=','))
        y_temp = np.array(pd.read_csv('train/' + train_y[i], sep=','))
        X_train = np.vstack((X_train, X_temp))
        y_train = np.vstack((y_train, y_temp))

    X_test = np.array(pd.read_csv('test/' + test_X[0], sep=','))
    y_test = np.array(pd.read_csv('test/' + test_y[0], sep=','))

    for i in range(1, len(test_X)):
        X_temp = np.array(pd.read_csv('test/' + test_X[i], sep=','))
        y_temp = np.array(pd.read_csv('test/' + test_y[i], sep=','))
        X_test = np.vstack((X_test, X_temp))
        y_test = np.vstack((y_test, y_temp))

    X = np.vstack([X_train, X_test])

    return X[:len(X_train),:], y_train.ravel(), X[len(X_train):,:], y_test.ravel()

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def standardization(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = ex1_data(('train1_icu_data.csv', 'train2_icu_data.csv'),\
        ('train1_icu_label.csv','train2_icu_label.csv'),\
        ('test1_icu_data.csv', 'test2_icu_data.csv'),\
        ('test1_icu_label.csv', 'test2_icu_label.csv'))
    
    pca = PCA(n_components='mle')
    newX = pca.fit_transform(X_train)
    k = pca.transform(X_train)
    print(newX[0])
    print(k[0])
