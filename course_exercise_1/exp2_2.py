from models import Perceptron
import data_loader
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

def plot_loss_curve(loss_history):
    x = []
    y = []

    for i, it in enumerate(loss_history):
        if i % 50 == 0:
            x.append(i)
            y.append(it)
    
    return x, y

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = data_loader.ex1_data(['train1_icu_data.csv', 'train2_icu_data.csv'],\
        ['train1_icu_label.csv','train2_icu_label.csv'],\
        ['test1_icu_data.csv', 'test2_icu_data.csv'],\
        ['test1_icu_label.csv', 'test2_icu_label.csv'])
    
    pca = PCA(n_components='mle')
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    
    clf_fir = Perceptron()
    train_history, validation_history, loss_history = clf_fir.fit(X_train, y_train, lr=1e-2, n_iter=int(1e3), n_splits=5, optimizer="BGD")
    train_history = np.mean(train_history, axis=0)
    validation_history = np.mean(validation_history, axis=0)
    loss_history = np.mean(loss_history, axis=0)
    
    x, y = plot_loss_curve(train_history)
    plt.subplot(1,2,1)
    plt.plot(x, y, label="BGD training score")
    x, y = plot_loss_curve(validation_history)
    plt.plot(x, y, label="BGD validation score")
    plt.legend()

    clf_vir = Perceptron()
    train_history, validation_history, loss_history = clf_vir.fit(X_train, y_train, lr=1e-2, n_iter=int(1e3), n_splits=5,optimizer="Adam")
    train_history = np.mean(train_history, axis=0)
    validation_history = np.mean(validation_history, axis=0)
    loss_history = np.mean(loss_history, axis=0)

    x, y = plot_loss_curve(train_history)
    plt.subplot(1,2,2)
    plt.plot(x, y, label="Adam training score")
    x, y = plot_loss_curve(validation_history)
    plt.plot(x, y, label="Adam validation score")
    plt.legend()
    plt.show()

    print(clf_fir.score(X_test, y_test))
    print(clf_vir.score(X_test, y_test))
