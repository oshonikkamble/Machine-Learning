import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn import datasets
from sklearn.svm import SVC

def SVM():
    cancer = datasets.load_breast_cancer()

    print("Features of the cancer dataset: ",cancer.feature_names)

    print("Labels of the cancer datasets: ",cancer.target_names)

    print("First 5 records are: ")
    print(cancer.data[0:5])

    print("target of dataset : ",cancer.target)

    X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target,test_size = 0.3,random_state = 109)

    clf = svm.SVC(kernel = "linear")
    clf.fit(X_train,y_train)
    y_predict= clf.predict(X_test)

    print("Accuracy of the model is : ",metrics.accuracy_score(y_test,y_predict)*100)

def main():
    print()

    print("------Support Vector Machine------")

    SVM()

if __name__ == "__main__":
    main()