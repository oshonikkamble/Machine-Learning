import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def HeadBrainPredictor():
    data = pd.read_csv("MarvellousHeadBrain.csv")
    
    print("Size if not",data.shape)

    X = data["Head Size(cm^3)"].values
    Y = data["Brain Weight(grams)"].values


    mean_x = np.mean(X)
    mean_y = np.mean(Y)

    n = len(X)




    numerator = 0
    denomentaor = 0


    for i in range(n):
        numerator = numerator + (X[i] - mean_x) * (Y[i] - mean_y)
        denomentaor = denomentaor + (X[i] - mean_x)**2

    m = numerator / denomentaor

    c = mean_y - (m * mean_x) 

    print("Slope of Regression line is ",m)
    print("Y Intersect of Regression line is ",c)

    max_x = np.max(X)+100
    min_x = np.min(X)-100


    x = np.linspace(min_x,max_x,n)
    y = c + m * x

    plt.plot(x,y, color = "#00CC00", labels = "Regression Line")
    plt.plot(x,y, color = "#0000CC", labels = "Scatter Plot")

    plt.xlabel("Head size in cm^3")
    plt.ylabel("Brain Weight in gram")

    plt.legend()
    plt.show()

    # Findout goodness of fit ie R Sqaure

    numerator = 0
    denomentaor = 0
    
    for i in range(n):
        y_predict = c + m * X[i]
        numerator = numerator + (Y[i] - mean_y)**2
        denomentaor = denomentaor + (Y[i] - y_predict) ** 2

    r2 = 1 - (denomentaor / numerator)
    print(r2)

def main():
    print("Supervised Machine Learning")
    print("")
    print("Linear Regression on Head and Brain size Data set")

    HeadBrainPredictor()

if __name__ == "__main__":
    main()