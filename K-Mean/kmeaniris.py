from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"C:\Users\Dell\Desktop\Kmean\Iriscase.csv")
x = dataset.iloc[:,[0,1,2,3]].values

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i,init ="k-means++",max_iter = 300 , n_init = 10,random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title("the elbow method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")  #within cluster sum of sqaures
plt.show()

# Apllying kmeans to the dataset / creatingthe kmeans classifier

kmeans = KMeans(n_clusters = 3,init = "k-means++", max_iter = 300,n_init = 10,random_state =0) 
y_kmeans = kmeans.fit_predict(x)

#visualising the cluster\

plt.scatter(x[y_kmeans ==0,0],x[y_kmeans == 0,1],s=100,c="red",label = "Iris-sentosa")
plt.scatter(x[y_kmeans ==1,0],x[y_kmeans == 1,1],s=100,c="blue",label = "Iris-versicolor")
plt.scatter(x[y_kmeans ==2,0],x[y_kmeans == 2,1],s=100,c="green",label = "Iris-verginika")


#pltoing the centroids of the cluster

plt.scatter(kmeans.cluster_centers_[:0],kmeans.cluster_centers_[:,1],s = 100,c="Yellow",label = "Centroids")

plt.legend()
plt.show()