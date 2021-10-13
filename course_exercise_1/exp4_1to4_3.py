import data_loader
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection  import cross_val_score

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = data_loader.ex1_data(['train1_icu_data.csv'],\
    ['train1_icu_label.csv'],\
    ['test1_icu_data.csv'],\
    ['test1_icu_label.csv'])

    k_range = range(1, 80)
    k_score = []

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
        scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
        print("[%d/%d] %.4f " % (k, len(k_range), scores.mean()))
        k_score.append(scores.mean())

    knn = KNeighborsClassifier(n_neighbors=23, weights='distance')
    knn.fit(X_train, y_train)
    print(knn.score(X_test, y_test))

    #画图，x轴为k值，y值为误差值
    plt.plot(k_range, k_score, label='weight=distance')
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Score')
    plt.legend()
    plt.show()