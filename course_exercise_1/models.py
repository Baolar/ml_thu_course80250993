import enum
import numpy as np
from numpy.lib.function_base import gradient
from numpy.random.mtrand import beta, rand
from sklearn.utils import validation
import data_loader
from tqdm import tqdm
import random
import math
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit

class Base_Model:
    def __init__(self):
        self.w = None
        self.b = None
        self.dict_y = dict()
        self.res_dict_y = dict()
    
    def predict(self, X):
        res = []
        for row in X:
            row_y = np.dot(row, self.w) + self.b
            if row_y > 0:
                res.append(self.res_dict_y[1])
            else:
                res.append(self.res_dict_y[-1])
            
        return res
    
    def score(self, X_test, y_test):
        res = self.predict(X_test)
        cnt = 0

        for i in range(len(X_test)):
            if res[i] == y_test[i]:
                cnt += 1
        
        return cnt / len(X_test)


class Fisher_Linear_Discriminant(Base_Model):
    def __init__(self):
        Base_Model.__init__(self)
    
    def fit(self, X, y):
        species = list(set(y))

        self.dict_y[species[0]] = 1
        self.res_dict_y[1] = species[0]
        if len(species) > 1:
            self.dict_y[species[1]] = -1
            self.res_dict_y[-1] = species[1]

        n = len(X)
        dim = len(X[0])
        w = np.zeros(dim)
        m1 = np.zeros(dim)
        m2 = np.zeros(dim)
        n1 = 0
        n2 = 0

        for i in range(n):
            if self.dict_y[y[i]] == 1:
                m1 += X[i]
                n1 += 1
            else:
                m2 += X[i]
                n2 += 1

        m = (m1 + m2) / n
        m1 /= n1
        m2 /= n2
        S1 = np.zeros((dim, dim))
        S2 = np.zeros((dim, dim))

        for i in range(n):
            if self.dict_y[y[i]] == -1:
                S1 += (X[i] - m1).reshape(dim, 1) * \
                    (X[i] - m1).reshape(1, dim)

            else:
                S2 += (X[i] - m2).reshape(dim, 1) * \
                    (X[i] - m2).reshape(1, dim)

        Sw = S1 + S2 

        w = np.dot(np.linalg.inv(Sw), (m1 - m2).reshape(dim, 1)).flatten()
        m1_t = np.dot(w, m1)
        m2_t = np.dot(w, m2)
        w0 = -0.5 * (m1_t + m2_t)
        w0 = -np.dot(w, m)
        
        self.w = w
        self.b = w0
   
    # f(x) = sign(w * x + b)
class Perceptron(Base_Model):
    def __init__(self):
        Base_Model.__init__(self)

    def sign(x):
        return 1 if x >= 0 else -1
    
    def fit(self, X, y, lr=1e-3, n_iter=1000, n_splits=5, optimizer='BGD'):
        self.w = np.zeros(len(X[0]))
        self.b = 0
        dim = len(X[0])

        species = list(set(y))
        self.dict_y[species[0]] = -1
        self.res_dict_y[-1] = species[0]

        if len(species) > 1:
            self.dict_y[species[1]] = 1
            self.res_dict_y[1] = species[1]

        # inialize for adam
        m = np.zeros(dim + 1)
        v = np.zeros(dim + 1)
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8


        # split dataset for cross-validation
        rs = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=0)

        # for each cross-validation record its history
        train_history = []
        validation_history = []
        loss_history = []

        cv_iter = 0
        for train_index, test_index in rs.split(X):
            cv_iter += 1
            rand_list = np.random.randn(dim + 1)
            w = rand_list[:dim]
            b = rand_list[-1]

            cv_train = []
            cv_validation = []
            cv_loss = []
            
            # each epoch
            for t in range(1, 1 + n_iter):
                loss = 0
                w_gradient = np.zeros(dim)
                b_gradient = 0

                train_cnt = 0
                # train
                for i in train_index:
                    g = np.dot(w, X[i]) + b
                    if g * self.dict_y[y[i]] <= 0:
                        w_gradient -= self.dict_y[y[i]] * X[i] #/ len(train_index)
                        b_gradient -= self.dict_y[y[i]] #/ len(train_index)
                        loss -= g * self.dict_y[y[i]]
                    else:
                        train_cnt += 1

                # loss /= len(train_index)

                # BGD
                if optimizer == "BGD":
                    w -= lr * w_gradient
                    b -= lr * b_gradient

                else:
                # Adam
                    m = beta1 * m + (1 - beta1) * np.hstack([w_gradient, b_gradient])
                    v = beta2 * v + (1 - beta2) * np.hstack([w_gradient, b_gradient]) * np.hstack([w_gradient, b_gradient])
                    m_ = m / (1 - beta1)
                    v_ = v / (1 - beta2)
                    theta = np.hstack([w, b]) - lr * m_ / (np.sqrt(v_) + eps)
                    w = theta[:dim]
                    b = theta[-1]

                validation_cnt = 0
                # cross-validation
                for i in test_index:
                    g = np.dot(w, X[i]) + b
                    if g * self.dict_y[y[i]] > 0:
                        validation_cnt += 1
            
                train_score = train_cnt / len(train_index)
                validation_score = validation_cnt /len(test_index)

                print("%d:[%d/%d] training score: %.4f\tvalidation score: %.4f\ttraining loss:%.2f"%
                    (cv_iter, t, n_iter, train_score, validation_score, np.float64(loss)))

                cv_train.append(train_score)
                cv_validation.append(validation_score)
                cv_loss.append(loss)

            train_history.append(cv_train)
            validation_history.append(cv_validation)
            loss_history.append(cv_loss)

            self.w += w
            self.b += b
        
        self.w /= n_splits
        self.b /= n_splits

        return train_history, validation_history, loss_history

# let const w0, b0 as the fist var.
# minimize sigma_(i=1)^n * log(1 + exp(- y_i(wTx + b)))
class Logistic_Regression(Base_Model):
    def __init__(self):
        Base_Model.__init__(self)

    def get_prob(self, X, y):
        print(self.res_dict_y[1])
        prob = []
        for i, xi in enumerate(X):
            # ans = self.res_dict_y[1] if np.dot(xi, self.w) + self.b > 0 else self.res_dict_y[-1]# xi > 0   1

            # if self.res_dict_y[1] == 
            res = 1 / (1 + np.exp(-(np.dot(self.w, xi) + self.b)))
            
            # if self.res_dict_y[1] == y[i]:
            prob.append(res)
           
        return prob

    
    def fit(self, X, y, lr=1e-2, n_iter=1000, n_splits=5, optimizer="BGD"):
        self.w = np.zeros(len(X[0]))
        self.b = 0
        dim = len(X[0])
        N = len(X)

        species = list(set(y))
        self.dict_y[species[0]] = -1
        self.res_dict_y[-1] = species[0]

        if len(species) > 1:
            self.dict_y[species[1]] = 1
            self.res_dict_y[1] = species[1]

        # inialize for adam
        m = np.zeros(dim + 1)
        v = np.zeros(dim + 1)
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8

        # split dataset for cross-validation
        rs = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=0)
        
        # for each cross-validation record its history
        train_history = []
        validation_history = []
        loss_history = []

        cv_iter = 0
        for train_index, test_index in rs.split(X):
            cv_iter += 1
            w = np.zeros(dim)
            b = 0

            cv_train = []
            cv_validation = []
            cv_loss = []
            
            for t in range(1, n_iter + 1):
                w_gradient = np.zeros(dim)
                b_gradient = 0
                loss = 0
                # train
                for i in train_index:
                    w_gradient -= self.dict_y[y[i]] * X[i] / (1 + np.exp(self.dict_y[y[i]] * (np.dot(w, X[i]) + b)))
                    b_gradient -= self.dict_y[y[i]] / (1 + np.exp(self.dict_y[y[i]] * (np.dot(w, X[i]) + b)))
                    loss += np.log(1 + np.exp(-y[i] * (np.dot(w, X[i]) + b)))

                loss /= N
                w_gradient /= N
                b_gradient /= N

                if optimizer == "SGD":
                    w -= lr * w_gradient
                    b -= lr * b_gradient
                else:
                    m = beta1 * m + (1 - beta1) * np.hstack([w_gradient, b_gradient])
                    v = beta2 * v + (1 - beta2) * np.hstack([w_gradient, b_gradient]) * np.hstack([w_gradient, b_gradient])
                    m_ = m / (1 - beta1)
                    v_ = v / (1 - beta2)
                    theta = np.hstack([w, b]) - lr * m_ / (np.sqrt(v_) + eps)
                    w = theta[:dim]
                    b = theta[-1]
                
                # calculate training score
                train_cnt = 0
                for i in train_index:
                    g = np.dot(w, X[i]) + b
                    if g * self.dict_y[y[i]] > 0:
                        train_cnt += 1
                
                validation_cnt = 0
                for i in test_index:
                    g = np.dot(w, X[i]) + b
                    if g * self.dict_y[y[i]] > 0:
                        validation_cnt += 1
                
                train_score = train_cnt / len(train_index)
                validation_score = validation_cnt / len(test_index)
             
                print("%d:[%d/%d] training score: %.4f\tvalidation score: %.4f\ttraining loss:%.2f"%
                    (cv_iter, t, n_iter, train_score, validation_score, np.float64(loss)))

                cv_train.append(train_score)
                cv_validation.append(validation_score)
                cv_loss.append(loss)

            train_history.append(cv_train)
            validation_history.append(cv_validation)
            loss_history.append(cv_loss)

            self.w += w
            self.b += b

        self.w /= n_splits
        self.b /= n_splits

        return train_history, validation_history, loss_history
            
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = data_loader.ex1_data(('train1_icu_data.csv', 'train2_icu_data.csv'),\
        ('train1_icu_label.csv','train2_icu_label.csv'),\
        ('test1_icu_data.csv', 'test2_icu_data.csv'),\
        ('test1_icu_label.csv', 'test2_icu_label.csv'))
    
    pca = PCA(n_components='mle')
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    
    clf = Logistic_Regression()
    train_history, validation_history, loss_history = clf.fit(X_train, y_train, lr=5e-3, n_iter=int(5e2), n_splits=5, optimizer="Adam")
    train_history = np.mean(train_history, axis=0)
    validation_history = np.mean(validation_history, axis=0)
    loss_history = np.mean(loss_history, axis=0)
    
    print(clf.score(X_test, y_test))
    # plt.plot(list(range(1, len(train_history) + 1)), train_history, label='train score')
    # plt.plot(list(range(1, len(validation_history) + 1)), validation_history, label='validation score')      
    plt.plot(list(range(1, len(loss_history) + 1)), loss_history, label='loss')   
    plt.legend()
    plt.show()

   


    # clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=30000, tol=1e-3))
    # clf.fit(X_train, y_train.ravel())
    # print(y_train[0:10])
    # print(y_train[0:10].ravel())
    # print(clf.score(X_test, y_test.ravel()))


        
        


