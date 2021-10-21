from numpy.core.numeric import cross
import data_loader
from sklearn.decomposition import PCA
from sklearn.model_selection import ShuffleSplit
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import time

def cross_validation(model, X_train):
    rs = ShuffleSplit(n_splits=5, test_size=0.1, random_state=0)
    cv = 0
    
    training_error = []
    cross_validation_error = []
    training_time = []
    for train_index, test_index in rs.split(X_train):
        cv += 1
        cv_X_train = X_train[train_index]
        cv_y_train = y_train[train_index]
        cv_X_test = X_train[test_index]
        cv_y_test = y_train[test_index]
        start_time_step = time.time()
        model.fit(cv_X_train, cv_y_train)
        stop_time_step = time.time()
        training_time.append(stop_time_step - start_time_step)
        training_error.append(1 - model.score(cv_X_train, cv_y_train))
        cross_validation_error.append(1 - model.score(cv_X_test, cv_y_test))
    
    return np.mean(training_error), np.mean(cross_validation_error), np.mean(training_time)
 

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
    X_test = scaler.transform(X_train)

    kernels = ['linear', 'rbf', 'poly']
    shrinking = [True, False]
    decision_function_shapes = ['ovo', 'ovr']
    Cs = [0.1, 1, 10]

    print("kernel\tshrinking\tC\tdecision_function_shape\ttrain_error\tcross validation error\ttraining time (ave)")
    for kernel in kernels:
        for is_shrinking in shrinking:
            for C in Cs:
                for decision_function_shape in decision_function_shapes:
                    model = SVC(C=C, kernel=kernel, shrinking=is_shrinking, decision_function_shape = decision_function_shape)
                    training_error, cv_error, training_time = cross_validation(model, X_train)
                    print("%s\t%s\t\t%s\t%s\t\t\t%.2f%%\t\t%.2f%%\t\t\t%.2f" % (kernel, is_shrinking, C, decision_function_shape, 100 * training_error, 100 * cv_error, training_time))

