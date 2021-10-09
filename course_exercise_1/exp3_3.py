import data_loader
from models import Logistic_Regression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import numpy as np

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = data_loader.ex1_data(['train1_icu_data.csv', 'train2_icu_data.csv'],\
    ['train1_icu_label.csv', 'train2_icu_label.csv'],\
    ['test1_icu_data.csv', 'test2_icu_data.csv'],\
    ['test1_icu_label.csv', 'test2_icu_label.csv'])
    pca = PCA(n_components="mle")
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    clf = Logistic_Regression()
    clf.fit(X_train, y_train, n_iter=500, lr=1e-2, n_splits=5, optimizer="Adam")
    prob = clf.get_prob(X_test, y_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, prob, pos_label=clf.res_dict_y[1])

    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()