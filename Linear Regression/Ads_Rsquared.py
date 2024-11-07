import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def Ads():

    data = pd.read_csv(r"C:\Users\Dell\Desktop\Advertising (1).csv")

    dataframe = pd.DataFrame(data)

    X = dataframe[["TV", "radio", "newspaper"]]
    y = dataframe["sales"]

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, )
    
    clf = LinearRegression()
    
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error: ",mse)
    print("R-squared: ",r2)

    plt.scatter(y_test, y_pred, color="blue")
    plt.plot(y_test, y_test, color="red", linestyle="--")
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Actual vs Predicted Sales")
    plt.show()

def main():

    print(" Linear Regression ")
    
    Ads()

if __name__ == "__main__":
    main()