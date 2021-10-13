from models import Logistic_Regression
import data_loader
from sklearn.decomposition import PCA
import csv
import numpy as np
import codecs

def write_predict(clf, path, X):
    with codecs.open(path, "w", "utf_8_sig") as csvfile:
        writer = csv.writer(csvfile)

        for row in X:
            writer.writerow(clf.predict([row]))


if __name__ == "__main__":
    X_train, y_train, X_test_1, y_test_1, X_test_2, y_test_2 = data_loader.final_data(['train1_icu_data.csv', 'train2_icu_data.csv'],\
    ['train1_icu_label.csv','train2_icu_label.csv'],\
    ['test1_icu_data.csv', 'test2_icu_data.csv'],\
    ['test1_icu_label.csv', 'test2_icu_label.csv'])
    pca = PCA(n_components="mle")
    X_train = pca.fit_transform(X_train, y_train)
    X_test_1 = pca.transform(X_test_1)
    X_test_2 = pca.transform(X_test_2)

    clf = Logistic_Regression()
    clf.fit(X_train, y_train, n_iter=int(1e3), lr=1e-3, n_splits=10, optimizer="Adam")
    print(clf.score(np.vstack((X_test_1, X_test_2)), np.hstack((y_test_1, y_test_2))))

    write_predict(clf, "test1_result.csv", X_test_1)
    write_predict(clf, "test2_result.csv", X_test_2)

        


    



