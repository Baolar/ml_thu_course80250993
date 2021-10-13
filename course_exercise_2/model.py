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
    return np.exp(X) / np.sum(np.exp(X), axis=-1, keepdims=True)

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


class ReLU(Layer):
    def __init__(self):
        Layer.__init__(self)
        self.x = None
    
    def __call__(self, X):
        self.x = X
        return self.__relu(X)

    def __relu(self, X):
        return np.maximum(0, X)
    
    def backward(self, grad):
        grad_x = np.array(self.x)
        grad_x[grad_x<=0] = 0.1
        grad_x[grad_x>0] = 1
        
        return grad_x * grad

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

        # inialize for adam
        self.m = np.zeros(self.W.shape)
        self.v = np.zeros(self.W.shape)
        self.m_b = np.zeros(self.b.shape)
        self.v_b = np.zeros(self.b.shape)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8

    def __call__(self, X):
        self.x = X
        return np.dot(X, self.W) + self.b
    
    def backward(self, grad):
        self.grad_W = np.outer(self.x, grad)
        self.grad_b = grad
        return np.dot(grad, np.transpose(self.W))
    
    def step(self, lr=1e-3):
        self.m = self.beta1 * self.m + (1 - self.beta1) * self.grad_W
        self.v = self.beta2 * self.v + (1 - self.beta2) * self.grad_W * self.grad_W
        m_ = self.m / (1 - self.beta1)
        v_ = self.v / (1 - self.beta2)
        self.W -= lr * m_ / (np.sqrt(v_) + self.eps)

        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * self.grad_b
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * self.grad_b * self.grad_b
        m_b_ = self.m_b / (1 - self.beta1)
        v_b_ = self.v_b / (1 - self.beta2)
        self.b -= lr * m_b_ / (np.sqrt(v_b_) + self.eps)

        # self.W -= lr * self.grad_W
        # self.b -= lr * self.grad_b

class Neural_Network:
    def __init__(self):
        self.linear1 = Linear(96, 32)
        self.sigmoid1 = Sigmoid()
        self.linear2 = Linear(16, 16)
        self.sigmoid2 = Sigmoid()
        self.linear3 = Linear(16, 2)
    
    def forward(self, X):
        X = self.linear1(X)
        X = self.sigmoid1(X)
        X = self.linear2(X)
        X = self.sigmoid2(X)
        X = self.linear3(X)

        return X

    def backward(self, grad):
        # grad_h = self.linear2.backward(grad)
        # grad_z_h = self.sigmoid1.backward(grad_h)
        # self.linear1.backward(grad_z_h)
        grad = self.linear3.backward(grad)
        grad = self.sigmoid2.backward(grad)
        grad = self.linear2.backward(grad)
        grad = self.sigmoid1.backward(grad)
        grad = self.linear1.backward(grad)
    
    def step(self, lr=1e-3):
        self.linear3.step(lr)
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



        
    
