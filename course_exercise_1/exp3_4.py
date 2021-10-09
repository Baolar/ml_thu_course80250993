import numpy as np
import data_loader
from sklearn.decomposition import PCA
import pandas as pd
import codecs
import csv

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = data_loader.ex1_data(['train1_icu_data.csv', 'train2_icu_data.csv'],\
    ['train1_icu_label.csv','train2_icu_label.csv'],\
    ['test1_icu_data.csv', 'test2_icu_data.csv'],\
    ['test1_icu_label.csv', 'test2_icu_label.csv'])

    pca = PCA(n_components="mle")
    t = pca.fit_transform(X_train, y_train)
    k1_spss = pca.components_.T 
    weight = (np.dot(k1_spss, pca.explained_variance_ratio_)) / np.sum(pca.explained_variance_ratio_)
    weighted_weight = weight / np.sum(weight)
    b = sorted(enumerate(weighted_weight), key=lambda x:x[1], reverse=True)

    description = data_loader.ex_feature_description()
  
    out_excel = []
    r = 1
    for index, weight in b:
        row = [r, weight, index+1]
        row.extend(description[index])
        out_excel.append(row)
        print("[%d] %.4f\t%d "% (r, weight, index+1), end="")
        print(description[index])
        r += 1

    print(out_excel)
    with codecs.open("feature_weight.csv", "w", "utf_8_sig") as csvfile:
        writer = csv.writer(csvfile)
        
        for row in out_excel:
            writer.writerow(row)
    



