import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from warnings import simplefilter

simplefilter(action = "ignore",category=FutureWarning)

print("Diabetes predictor using logistic Regression ")

diabete = pd.read_csv("diabetes.csv")

print("Columns of Dataset")
print(diabete.columns)

print("First 5 Records of Dataset")
print(diabete.head())

print("Dimension of diabetes data : {}".format(diabete.shape))

X_train ,X_Test,y_train,y_test = train_test_split(diabete.loc[:,diabete.columns != "Outcome"],diabete["Outcome"],stratify = diabete["Outcome"],random_state = 66)

logreg = LogisticRegression().fit(X_train,y_train)

print("Training set Accuracy :{:.3f}".format(logreg.score(X_train,y_train)))

print("test set accuracy :{:.3f}".format(logreg.score(X_Test,y_test)))

logreg001 = LogisticRegression(C=0.01).fit(X_train,y_train)

print("Training set Accuracy :{:.3f}".format(logreg001.score(X_train,y_train)))

print("Test set accuracy :{:.3f}".format(logreg001.score(X_Test,y_test)))

