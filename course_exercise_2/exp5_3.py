from model import *
import data_loader
from sklearn.decomposition import PCA
from sklearn.model_selection import ShuffleSplit
from model import train
from model import test
import numpy as np
import matplotlib.pyplot as plt

class Super_Neural_Network(Neural_Network):
    def __init__(self):
        Neural_Network.__init__(self)
        self.linear1 = Linear(96, 32)
        self.relu1 = ReLU()
        self.linear2 = Linear(32, 32)
        self.relu2 = ReLU()
        self.linear3 = Linear(32, 2)

    def forward(self, X):
        X = self.linear1(X)
        X = self.relu1(X)
        X = self.linear2(X)
        X = self.relu2(X)
        X = self.linear3(X)

        return X
    
    def backward(self, grad):
        grad = self.linear3.backward(grad)
        grad = self.relu2.backward(grad)
        grad = self.linear2.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.linear1.backward(grad)
    
    def step(self, lr=1e-3):
        self.linear3.step(lr)
        self.linear2.step(lr)
        self.linear1.step(lr)

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = data_loader.ex1_data(['train1_icu_data.csv'],\
    ['train1_icu_label.csv'],\
    ['test1_icu_data.csv'],\
    ['test1_icu_label.csv'])
    pca = PCA(n_components=96)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    epochs = 1500
    model = Super_Neural_Network()
    for epoch in range(1, epochs + 1):
        t_loss, t_acc = train(model, X_train, y_train, 1e-2)
        print("epoch:[%d/%d] %.2f %.4f%%" % (epoch, epochs, t_loss, 100 * t_acc))
    print(test(model, X_test, y_test))
