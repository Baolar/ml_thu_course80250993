from models import Fisher_Linear_Discriminant
from sklearn.decomposition import PCA
import data_loader
import matplotlib.pyplot as plt

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = data_loader.ex1_data(['train1_icu_data.csv'],\
        ['train1_icu_label.csv'],\
        ['test1_icu_data.csv'],\
        ['test1_icu_label.csv'])
    
    clf = Fisher_Linear_Discriminant()
    pca = PCA(n_components='mle')
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    clf.fit(X_train, y_train)
    print("error rate = %.2f%%"% (100 * (1 - clf.score(X_test, y_test))))
