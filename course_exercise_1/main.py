import numpy as np
from numpy.lib.npyio import load
import pandas as pd
from numpy.core.numerictypes import sctype2char
from sklearn import svm
import data_loader
from sklearn.decomposition import PCA

# Discriminant (FLD), Perceptron and 
# Logistic Regression (LR) algorithms in Python, 
# without directly importing packages of these algorithms. 
# Find a Python package of k-nearest neighbor (KNN) method 
# in scikit-learn or other packages. 
# Make observations the learning procedures, 
# performances and effects of optional choices on the performance.
def FLD(X, y):
    if len(X) < 1:
        return 0

    n = len(X)
    dim = len(X[0])
    w = np.zeros(dim)
    m1 = np.zeros(dim)
    m2 = np.zeros(dim)
    n1 = 0
    n2 = 0

    for i in range(n):
        if y[i] == 0:
            m1 += X[i]
            n1 += 1
        else:
            m2 += X[i]
            n2 += 1

    m = (m1 + m2) / n
    m1 /= n1
    m2 /= n2
    S1 = np.zeros((dim, dim))
    S2 = np.zeros((dim, dim))

    for i in range(n):
        if y[i] == 0:
            S1 += (X[i] - m1).reshape(dim, 1) * \
                (X[i] - m1).reshape(1, dim)
            # S1 += np.dot((X[i] - m1).T, X[i] - m1)
        else:
            S2 += (X[i] - m2).reshape(dim, 1) * \
                (X[i] - m2).reshape(1, dim)
            # S2 += np.dot((X[i] - m2).T, X[i] - m2)

    Sw = S1 + S2 

    w = np.dot(np.linalg.inv(Sw), (m1 - m2).reshape(dim, 1)).flatten()
    m1_t = np.dot(w, m1)
    m2_t = np.dot(w, m2)
    w0 = -0.5 * (m1_t + m2_t)
    w0 = -np.dot(w, m)
    return w, w0

def load_data():
    X1 = np.array(pd.read_csv('dataset/train1_icu_data.csv', sep=','))
    y1 = np.array(pd.read_csv('dataset/train1_icu_label.csv', sep=','))
    X2 = np.array(pd.read_csv('dataset/train2_icu_data.csv', sep=','))
    y2 = np.array(pd.read_csv('dataset/train2_icu_label.csv', sep=','))
    X_train = np.vstack((X1, X2))
    y_train = np.vstack((y1, y2))

    X_test1 = np.array(pd.read_csv('test/test1_icu_data.csv', sep=','))
    y_test1 = np.array(pd.read_csv('test/test1_icu_label.csv', sep=','))
    X_test2 = np.array(pd.read_csv('test/test2_icu_data.csv', sep=','))
    y_test2 = np.array(pd.read_csv('test/test2_icu_label.csv', sep=','))
    X_test = np.vstack([X_test1, X_test2])
    y_test = np.vstack([y_test1, y_test2])

    return X_train, y_train, X_test, y_test

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def predict(X, w, w0):
    gx = np.dot(X, w) + w0
    y = []
    for g in gx:
        if g >= 0:
            y.append(0)
        else:
            y.append(1)
    
    return np.array(y)

if __name__ == "__main__":
    # X = [[1,1,1],[2,3,4],[2,1,2],[3,1,0],[213,21,31],[341,23,2]]
    # y = [0, 1, 0, 1, 0, 1]
    # print(FLD(X, y))
    # X_train, y_train, X_test, y_test = data_loader.ex1()
    X_train, y_train, X_test, y_test = data_loader.ex1_data(['train1_icu_data.csv'],\
        ['train1_icu_label.csv'],\
        ['test1_icu_data.csv'],\
        ['test1_icu_label.csv'])
    pca = PCA(n_components='mle')
    X_train = pca.fit_transform(X_train) 
    X_test = pca.transform(X_test)
    w, w0 = FLD(X_train, y_train)

    y = predict(X_test, w, w0)

    cnt = 0
    for i in range(len(y)):
        if y_test[i] == y[i]:
            cnt += 1
        # print(str(y_test[i]) + " " + str(y[i]))
    print(cnt / len(y))

    # cnt = 0
    # clf = svm.SVC()
    # clf.fit(X_train[:,:37], y_train)
    # y = clf.predict(X_test[:,:37])
    # for i in range(len(y)):
    #     if y_test[i] == y[i]:
    #         cnt += 1
    # print(cnt / len(y))


    



