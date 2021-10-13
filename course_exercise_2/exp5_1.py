from model import Neural_Network
import data_loader
from sklearn.decomposition import PCA
from sklearn.model_selection import ShuffleSplit
from model import train
from model import test
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = data_loader.ex1_data(['train1_icu_data.csv'],\
    ['train1_icu_label.csv'],\
    ['test1_icu_data.csv'],\
    ['test1_icu_label.csv'])
    pca = PCA(n_components=96)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    rs = ShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    cv = 0
    epochs = 3000

    train_loss_history_plot = []
    train_acc_history_plot = []
    test_acc_history_plot = []


    for train_index, test_index in rs.split(X_train):
        cv += 1
        model = Neural_Network()
        train_loss_history = []
        train_acc_history = []
        test_acc_history = []

        for epoch in range(1, epochs + 1):
            train_loss = 0
            train_acc = 0
            for index in train_index:
                t_loss, t_acc = train(model, np.atleast_2d(X_train[index]), np.atleast_1d(y_train[index]),lr=1e-2)
                train_loss += t_loss
                train_acc += t_acc
            train_loss /= len(train_index)
            train_acc /= len(train_index)
            print("cv:%d epoch:[%d/%d] %.2f %.4f%%" % (cv, epoch, epochs, train_loss, 100 * train_acc))
            train_loss_history.append(train_loss)
            train_acc_history.append(train_acc)

            if epoch % 200 == 0:
                test_acc = 0
                for index in test_index:
                    test_acc += test(model, np.atleast_2d(X_train[index]), np.atleast_1d(y_train[index]))
                test_acc /= len(test_index)
                test_acc_history.append(test_acc)
                print("cross-validation: %d %.4f%%" % (cv, 100 * test_acc))
        train_loss_history_plot.append(train_loss_history)
        train_acc_history_plot.append(train_acc_history)
        test_acc_history_plot.append(test_acc_history)
    train_loss_history_plot = np.mean(train_loss_history_plot, axis=0)
    train_acc_history_plot = np.mean(train_acc_history_plot, axis=0)
    test_acc_history_plot = np.mean(test_acc_history_plot, axis=0)

    print(test(model, X_test, y_test))

    plt.subplot(2,1,1)
    plt.plot(list(range(1, 1 + len(train_loss_history_plot))), train_loss_history_plot, label="loss")
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(list(range(1, 1 + len(train_acc_history_plot))), train_acc_history_plot, label="train acc")
    plt.plot(200 *  np.array(range(1, 1 + len(test_acc_history_plot))), test_acc_history_plot, label="test acc")
    plt.legend()
    plt.show()
