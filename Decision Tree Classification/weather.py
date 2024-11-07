from sklearn import tree
import numpy
import pandas


def MarvellousClassifier(weather,temperature):
    Features = {["Sunny : Hot"],["Sunny : Hot"],["Overcast:Hot"],["Rainy:Mild"],["Rainy:Cool"],["Rainy:Cool"],["Overcast:Cool"],["Sunny:Mild"],["Sunny:Cool"],["Rainy:Mild"]}

    Labels = [0,0,1,1,1,0,1,0,1,1]

    obj = tree.DecisionTreeClassifier()
    obj = obj.fit(Features,Labels)

    ret = obj.predict([[weather,temperature]])

    if ret == 3:
        print("Yes You Can Play")

    else:
        print("No You Cant Play")

def main():

    print("Please Enter the Information About Play Prediction")

    print("Please enter Weather Condition Sunny / Rainy / Overcast")

    Weather = input()

    if Weather.lower() == "Sunny":
        Weather == 1

    elif Weather.lower() == "Rainy":
        Weather == 2
    
    elif Weather.lower() == "Overcast":
        Weather == 3
    else:
        print("Invalid Type of Surface")

        exit()

    print("Please Enter Temperature Hot / Mild / Cool")

    Temp = input()

    if Temp.lower() == "Hot":
        Temp == 1

    elif Temp.lower() == "Mild":
        Temp == 2
    
    elif Temp.lower() == "Cool":
        Temp == 3
    else:
        print("Invalid Type of Surface")

        exit()

    MarvellousClassifier(Weather,Temp)
if __name__ == "__main__":
    main()
