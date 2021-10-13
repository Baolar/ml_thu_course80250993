import numpy as np
import data_loader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
def one_hot(n_classes,y):
    return np.eye(n_classes)[y]

def nll(Y_true,Y_pred):
    Y_true=np.atleast_2d(Y_true)
    Y_pred=np.atleast_2d(Y_pred)
    loglikelihoods = np.sum(np.log(EPSILON + Y_pred) * Y_true, axis=-1)
    return -np.mean(loglikelihoods)

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def softmax(X):
    """
    这种定义方法可以同时处理向量和矩阵
    """
    exp = np.exp(X)
    return exp/np.sum(exp,axis=-1,keepdims=True)


def dsigmoid(X):
    sig=sigmoid(X)
    return sig * (1 - sig)

EPSILON=1e-8

class NeuralNet():
    """只有一层隐藏的MLP"""
    def __init__(self,input_size,hidden_size,output_size):
        self.W_h=np.random.uniform(size=(input_size,hidden_size),high=0.01,low=-0.01)
        self.b_h=np.zeros(hidden_size)
        self.W_o = np.random.uniform(size=(hidden_size, output_size), high=0.01, low=-0.01)
        self.b_o = np.zeros(output_size)
        self.output_size = output_size
    def forward(self,X,keep_activations=False):
        z_h=np.dot(X,self.W_h)+self.b_h
        h = sigmoid(z_h)
        z_o = np.dot(h, self.W_o) + self.b_o
        y = softmax(z_o)
        if keep_activations:
            return y, h, z_h
        else:
            return y
    def loss(self, X, y):
        return nll(one_hot(self.output_size, y), self.forward(X))
    
    def grad_loss(self, x, y_true):
        y, h, z_h = self.forward(x, keep_activations=True)
        grad_z_o = y - one_hot(self.output_size, y_true)

        grad_W_o = np.outer(h, grad_z_o)
        grad_b_o = grad_z_o
        grad_h = np.dot(grad_z_o, np.transpose(self.W_o))
        grad_z_h = grad_h * dsigmoid(z_h)
        grad_W_h = np.outer(x, grad_z_h)
        grad_b_h = grad_z_h
        grads = {"W_h": grad_W_h, "b_h": grad_b_h,
                 "W_o": grad_W_o, "b_o": grad_b_o}
        return grads
    def train(self, x, y, learning_rate):
        # Traditional SGD update on one sample at a time
        grads = self.grad_loss(x, y)
        self.W_h = self.W_h - learning_rate * grads["W_h"]
        self.b_h = self.b_h - learning_rate * grads["b_h"]
        self.W_o = self.W_o - learning_rate * grads["W_o"]
        self.b_o = self.b_o - learning_rate * grads["b_o"]

    def predict(self, X):
        if len(X.shape) == 1:
            return np.argmax(self.forward(X))
        else:
            return np.argmax(self.forward(X), axis=1)

    def accuracy(self, X, y):
        y_preds = np.argmax(self.forward(X), axis=1)
        return np.mean(y_preds == y)

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = data_loader.ex1_data(['train1_icu_data.csv'],\
    ['train1_icu_label.csv'],\
    ['test1_icu_data.csv'],\
    ['test1_icu_label.csv'])
    pca = PCA(n_components=96)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    n_hidden = 32
    model = NeuralNet(96, n_hidden, 2)
    
    losses = []
    accuracies = []
    accuracies_test = []
    for epoch in range(50000):
        for i, (x, y) in enumerate(zip(X_train, y_train)):
            model.train(x, y, 0.001)

        losses.append(model.loss(X_train, y_train))
        accuracies.append(model.accuracy(X_train, y_train))
        accuracies_test.append(model.accuracy(X_test, y_test))
        print("Epoch #%d, train loss: %0.5f, train acc: %0.3f, test acc: %0.3f"
            % (epoch + 1, losses[-1], accuracies[-1], accuracies_test[-1]))
    
    plt.plot(losses)
    plt.title("Training loss")
    plt.show()
