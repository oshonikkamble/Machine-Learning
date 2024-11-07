from sklearn import tree   

def MarvellousClassifier(weight,surface):

    Features = [[35,1],[4,71], [90,0], [48,1], [90,0], [35,1] ,[92,1] ,[35,1] ,[35,1], [35,1] ]


    Labels = [1,1,2,1,2,1,2,1,1,1]

    obj = tree.DecisionTreeClassifier()

    obj = obj.fit(Features,Labels)

    ret = obj.predict([[weight,surface]])
    if ret == 1:
        print("Your Object is looklike Tennis ball")

    else:
        print("Your Object is looklike Cricket ball")


def main():

    print("------- Ball type classification case study------")

    print("Please enter the information about the object that you want to test")

    print("Please enter the weight of your object in grams")
    no = int(input())

    print("please mention the type of surface rough / Smooth")
    data= input()

    if data.lower() == "rough":
        data == 1
    elif data.lower() == "smooth":
        data == 0
    else:
        print("Invalid type of surface")

        exit()


    MarvellousClassifier(no,data)

if __name__ == "__main__":
    main()