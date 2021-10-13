from numpy import random
import data_loader
import numpy as np
from sklearn.decomposition import PCA

def loos_fn(p, y):
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))

def loss_grad(p, y):
    return -(y / p + (y - 1) / (1 - p))

class Layer:
    def __init__(self, input_size, output_size):
        self.x = None
    def backward(self, grad):
        pass
    def step(self, lr=1e-3):
        pass

class Linear(Layer):
    def __init__(self, input_size, output_size):
        Layer.__init__(self, input_size, output_size)
        self.deltaW = np.zeros((output_size, input_size))
        self.W = np.random.rand(output_size, input_size)
    
    def __call__(self, x):
        self.x = x
        return np.dot(self.W, x)
    
    def backward(self, next_backward):
        self.deltaW = np.dot(self.x.reshape(-1, 1), next_backward.reshape((1, -1)))
        return np.dot(self.W.T, next_backward.reshape(-1, 1))
    
    def step(self, lr=1e-3):
        self.W -= lr * self.deltaW.reshape(self.W.shape)
    
    def zero_grad(self):
        self.deltaW = np.zeros(self.W.shape)

class ReLU(Layer):
    def __init__(self, input_size):
        Layer.__init__(self, input_size, input_size)
    
    def __call__(self, x):
        self.x = x
        y = np.maximum(0, x)
        y[0] = x[0]
        return y
    
    def backward(self, next_backward):
        pre_backward = np.zeros(len(next_backward))
        for i, xi in enumerate(self.x):
            if xi > 0 or i == 1:
                pre_backward[i] = 0
        pre_backward *= next_backward.ravel()

        return pre_backward
    
# 0 survived
# 1 dead
class ML_Perceptron():
    def __init__(self):
        self.linear1 = Linear(97, 32)
        self.relu1 = ReLU(32)
        self.linear2 = Linear(32, 32)
        self.relu2 = ReLU(32)
        self.linear3 = Linear(32, 1)

    def forward(self, x):
        x = self.linear1(x)
        #print(x)
        x = self.relu1(x)
        #print(x)
        x = self.linear2(x)
        #print(x)
        x = self.relu2(x)
        #print(x)
        x = self.linear3(x)
        #print(x)
        #print("===============")
        return x
    
    def backward(self, grad):
        grad = np.array(grad).reshape((-1, 1))
        grad = self.linear3.backward(grad)
        grad = self.relu2.backward(grad)
        grad = self.linear2.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.linear1.backward(grad)
    
    def zero_grad(self):
        self.linear1.zero_grad()
        self.linear2.zero_grad()
        self.linear3.zero_grad()

    def step(self, lr=1e-2):
        self.linear3.step(lr)
        self.linear2.step(lr)
        self.linear1.step(lr)

def train(model, X, y, lr=1e-2):
    N = len(X)
    loss = 0
    w_grad = 0
    for i in range(N):
        pre = model.forward(X[i])
        if y[i] * pre <= 0:
            #print("classify error! %d %f" % (y[i], pre))
            w_grad -= y[i] * X[i]
            loss -= y[i] * pre
            #print("%d  %f"%(y[i],pre))
    print("loss:%f"%(loss))
    model.zero_grad()
    model.backward(loss)
    model.step(lr)

def test(model, X, y):
    N = len(X)
    cnt = 0
    for i in range(N):
        pre = model.forward(X[i])
        if y[i] * pre > 0:
            cnt += 1
            
    return cnt / N
        
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = data_loader.ex1_data(['train1_icu_data.csv'],\
    ['train1_icu_label.csv'],\
    ['test1_icu_data.csv'],\
    ['test1_icu_label.csv'])

    for i in range(len(y_train)):
        if y_train[i] == 0:
            y_train[i] = -1
    for i in range(len(y_test)):
        if y_test[i] == 0:
            y_test[i] = -1
    
    pca = PCA(n_components="mle")
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    
    X_train = np.insert(X_train, 0, values=1, axis=1)
    X_test = np.insert(X_test, 0, values=1, axis=1)

    epochs = 1
    model = ML_Perceptron()
    for i in range(epochs):
        train(model, X_train, y_train, lr=1e-4)
    
    print(model.forward(X_train[1]))
    # print(test(model, X_test, y_test))