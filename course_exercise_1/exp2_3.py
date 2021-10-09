from models import Perceptron
import data_loader
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    X_train, y_train, X_test_1, y_test_1 = data_loader.ex1_data(['train2_icu_data.csv'],\
        ['train2_icu_label.csv'],\
        ['test1_icu_data.csv'],\
        ['test1_icu_label.csv'])

    X_train, y_train, X_test_2, y_test_2 = data_loader.ex1_data(['train2_icu_data.csv'],\
        ['train2_icu_label.csv'],\
        ['test2_icu_data.csv'],\
        ['test2_icu_label.csv'])
    
    pca = PCA(n_components='mle')
    X_train = pca.fit_transform(X_train)
    X_test_1 = pca.transform(X_test_1)
    X_test_2 = pca.transform(X_test_2)
    clf = Perceptron()
    clf.fit(X_train, y_train, lr=5e-3, n_iter=int(3e3), n_splits=5,optimizer="Adam")
    
    print("error rate: %.2f%%" % (100 * (1 - clf.score(X_test_1, y_test_1))))
    print("error rate: %.2f%%" % (100 * (1 - clf.score(X_test_2, y_test_2))))

 
