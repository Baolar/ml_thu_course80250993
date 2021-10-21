import data_loader
from sklearn.decomposition import PCA
from sklearn.model_selection import ShuffleSplit
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import time
import csv
import codecs


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = data_loader.ex1_data(['train1_icu_data.csv'],\
    ['train1_icu_label.csv'],\
    ['test1_icu_data.csv'],\
    ['test1_icu_label.csv'])
    pca = PCA(n_components=96)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
 
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = SVC(C=1, kernel='linear')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    with codecs.open("submissions.csv", "w", "utf_8_sig") as csvfile:
        writer = csv.writer(csvfile)
        for row in y_pred:
            writer.writerow([row])
    
    print(model.score(X_test, y_test))
