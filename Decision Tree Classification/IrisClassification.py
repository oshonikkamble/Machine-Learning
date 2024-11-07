from sklearn import tree
from sklearn.datasets import load_iris  
from sklearn.model_selection import train_test_split
from sklearn.model_selection import accuracy_score
def main():    
    print("------ Iris Flower Classification Case Study -----")

    iris = load_iris()  #150 * 5
    
    #print(data)
    #print(shape(iris))
    print(iris)
    
    
    
    Features = iris.data #150 * 4(column)
    Labels = iris.target # 150 * 1

    print("Features Are : ")
    print(Features)

    print("Labels Are : ")
    print(Labels)

    # Focus to execute this:
    data_train , data_test , target_train , target_test = train_test_split(Features, Labels, test_size = 0.5)
    #75*4           75*4        75*1            75*1                   
                                                                                                                                 
    obj=tree.DecisionTreeClassifier()

    obj = obj.fit(data_train,target_train)

    output = obj.predict(data_test)

    accuracy = accuracy_score(target_test , output)

    print("ACCURACY IS : " ,accuracy * 100,"%" )


if __name__ == "__main__":
    main()
