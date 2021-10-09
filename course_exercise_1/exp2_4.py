from models import Perceptron
import data_loader
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = data_loader.ex1_data(['train1_icu_data.csv'],\
        ['train1_icu_label.csv'],\
        ['test1_icu_data.csv'],\
        ['test1_icu_label.csv'])

    pca = PCA(n_components='mle')
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    clf = Perceptron()
    clf.fit(X_train, y_train, lr=5e-3, n_iter=int(3e3), n_splits=5,optimizer="Adam")
    
    print("error rate: %.2f%%"% (100 * (1 - clf.score(X_test, y_test))))

 
