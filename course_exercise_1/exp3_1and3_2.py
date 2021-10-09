from models import Logistic_Regression
import data_loader
from sklearn.decomposition import PCA
import numpy as np

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = data_loader.ex1_data(['train1_icu_data.csv'],\
    ['train1_icu_label.csv'],\
    ['test1_icu_data.csv'],\
    ['test1_icu_label.csv'])
    pca = PCA(n_components="mle")
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    clf = Logistic_Regression()
    train_history, validation_history, loss_hitory = clf.fit(X_train, y_train, n_iter=1000, lr=1e-2, n_splits=5, optimizer="Adam")
    print("training error: %.4f\tvalidation error: %.4f\ttest error: %.4f" % (1-np.mean(train_history, axis=0)[-1],\
        1-np.mean(validation_history, axis=0)[-1], 1-clf.score(X_test, y_test)))