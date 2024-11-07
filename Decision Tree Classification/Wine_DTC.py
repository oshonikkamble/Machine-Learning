from sklearn import datasets
from sklearn.datasets import load_wine
from sklearn import tree
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def Winepredict():
    wine = load_wine()
    print(wine.feature_names)

    print(wine.target_names)

    print(wine.data[0:5])

    print(wine.target)

    x = wine.data
    y = wine.target

    features = wine.feature_names
    X_train , X_test , Y_train , Y_test = train_test_split(x ,y, test_size=0.5)

    clf = DecisionTreeClassifier()

    clf.fit(X_train , Y_train)

    y_predict = clf.predict(X_test)

    Accuracy = accuracy_score(Y_test,y_predict)

    print("Accuracy :  ",Accuracy % 100 )

    plt.scatter(Y_test, y_predict, color="blue")
    plt.plot(Y_test, Y_test, color="red", linestyle="--")
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Actual vs Predicted Sales")
    plt.legend()
    plt.show()  


def main():

    print("Decision Tree Classifier")

    print()

    Winepredict()

if __name__ == "__main__":
    main()