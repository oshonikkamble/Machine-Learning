from sklearn import tree
import numpy as np

def MarvellousClassifier(weather, temperature):
    Features = [[1, 1], [1, 1], [3, 1], [2, 2], [2, 3], [2, 3], [3, 3], [1, 2], [1, 3], [2, 2]]
    Labels = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1]
    obj = tree.DecisionTreeClassifier()
    obj = obj.fit(Features, Labels)
    ret = obj.predict([[weather, temperature]])
    if ret[0] == 1:
        print("Yes You Can Play")
    else:
        print("No You Can't Play")

def main():
    print("Please Enter the Information About Play Prediction")
    print("Please enter Weather Condition Sunny / Rainy / Overcast")
    Weather = input()

    if Weather.lower() == "sunny":
        Weather = 1
    elif Weather.lower() == "rainy":
        Weather = 2
    elif Weather.lower() == "overcast":
        Weather = 3
    else:
        print("Invalid Type of Weather")
        exit()
    print("Please Enter Temperature Hot / Mild / Cool")
    Temp = input()

    if Temp.lower() == "hot":
        Temp = 1
    elif Temp.lower() == "mild":
        Temp = 2
    elif Temp.lower() == "cool":
        Temp = 3
    else:
        print("Invalid Type of Temperature")
        exit()

    MarvellousClassifier(Weather, Temp)

if __name__ == "__main__":
    main()

