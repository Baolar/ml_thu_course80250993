import numpy as np
from torch._C import TensorType
import data_loader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import ShuffleSplit

EPSILON = 1e-8

def one_hot(n_classes, y):
    return np.eye(n_classes)[y]

def nll(Y_true, Y_pred):
    Y_pred = softmax(Y_pred)
    Y_true = np.atleast_2d(Y_true)
    Y_pred = np.atleast_2d(Y_pred)
    loglikelihoods = np.sum(np.log(EPSILON + Y_pred) * Y_true, axis=-1)

    return -np.mean(loglikelihoods)

def softmax(X):
    return np.exp(X) / np.sum(np.exp(X), axis=-1,keepdims=True)

class Layer:
    def __init__(self):
        pass

class Sigmoid(Layer):
    def __init__(self):
        Layer.__init__(self)
        self.x = None

    def __call__(self, X):
        self.x = X
        return self.__sigmoid(X)

    def __sigmoid(self, X):
        return 1 / (1 + np.exp(-X))
    
    def backward(self, grad):
        return grad * self.__sigmoid(self.x) * (1 - self.__sigmoid(self.x))

class Linear(Layer):
    def __init__(self, input_size, output_size):
        Layer.__init__(self)
        self.x = None
        self.W = np.random.uniform(size=(input_size, output_size), high=1e-2, low=1e-2)
       #  self.W = np.random.rand(input_size, output_size)
        self.b = np.zeros(output_size)
        self.output_size = output_size

        self.grad_W = None
        self.grad_b = None

    def __call__(self, X):
        self.x = X
        return np.dot(X, self.W) + self.b
    
    def backward(self, grad):
        self.grad_W = np.outer(self.x, grad)
        self.grad_b = grad
        return np.dot(grad, np.transpose(self.W))
    
    def step(self, lr=1e-3):
        self.W -= lr * self.grad_W
        self.b -= lr * self.grad_b

class Neural_Network:
    def __init__(self):
        self.linear1 = Linear(96, 32)
        self.sigmoid1 = Sigmoid()
        self.linear2 = Linear(32, 2)
    
    def forward(self, X):
        X = self.linear1(X)
        X = self.sigmoid1(X)
        X = self.linear2(X)

        return X

    def backward(self, grad):
        grad_h = self.linear2.backward(grad)
        grad_z_h = self.sigmoid1.backward(grad_h)
        self.linear1.backward(grad_z_h)
    
    def step(self, lr=1e-3):
        self.linear2.step(lr)
        self.linear1.step(lr)
    
    def predict(self, X):
        if len(X.shape) == 1:
            return np.argmax(self.forward(X))
        else:
            return np.argmax(self.forward(X), axis=1)

    def accuracy(self, X, y):
        y_preds = np.argmax(self.forward(X), axis=1)
        return np.mean(y_preds==y)

def L_grad(model_out, y_true):
    prob = softmax(model_out)
    # print(prob)
    return one_hot(2, y_true) - prob

def train(model, X_train, y_train, lr=1e-3):
    loss = 0
    for i, (x, y) in enumerate(zip(X_train, y_train)):
        model_out = model.forward(x)
        loss += nll(one_hot(len(model_out), y), model_out.copy())
        # print(model_out)
        grad = -L_grad(model_out, y)
        # print(grad)
        model.backward(grad)
        model.step(lr)
    loss /= len(X_train)
    return loss, model.accuracy(X_train, y_train)

def test(model, X_test, y_test):
    return model.accuracy(X_test, y_test)

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = data_loader.ex1_data(['train1_icu_data.csv'],\
    ['train1_icu_label.csv'],\
    ['test1_icu_data.csv'],\
    ['test1_icu_label.csv'])
    pca = PCA(n_components=96)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    # model = Neural_Network()

    # epochs = 5000
    # for epoch in range(1, epochs + 1):
    #     l, a = train(model, X_train, y_train)
    #     b = test(model, X_test, y_test)
    #     print("epoch : [%d/%d]\tloss: %f\ttrain acc: %f\ttest acc: %f" % (epoch, epochs,l, a, b))
    
    # exit()
    rs = ShuffleSplit(n_splits=5, test_size=0.1, random_state=0)
    cv = 0
    epochs = 1000
    for train_index, test_index in rs.split(X_train):
        cv += 1
        model = Neural_Network()
        for epoch in range(1, epochs + 1):
            train_loss = 0
            train_acc = 0
            for index in train_index:
                t_loss, t_acc = train(model, np.atleast_2d(X_train[index]), np.atleast_1d(y_train[index]))
                train_loss += t_loss
                train_acc += t_acc
            train_loss /= len(train_index)
            train_acc /= len(train_index)
            print("cv:%d epoch:[%d/%d] %.2f %.4f%%" % (cv, epoch, epochs, train_loss, 100 * train_acc))

            if epoch % 200 == 0:
                test_acc = 0
                for index in test_index:
                    test_acc += test(model, np.atleast_2d(X_train[index]), np.atleast_1d(y_train[index]))
                test_acc /= len(test_index)
                print("cross-validation: %d %.4f%%" % (cv, 100 * test_acc))




        
    