#Importing required modules
import time
import numpy as np
from scipy.spatial.distance import cdist 
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
 
#Defining our function 
# assigns, comparisons, executed lines
def kmeans(x,k, no_of_iterations, measurementData):
    idx = np.random.choice(len(x), k, replace=False)
    #Randomly choosing Centroids 
    centroids = x[idx, :] #Step 1
     
    #finding the distance between centroids and all the data points
    distances = cdist(x, centroids ,'euclidean') #Step 2
     
    #Centroid with the minimum Distance
    points = np.array([np.argmin(i) for i in distances]) #Step 3

    # measurement
    measurementData[0] += 4
    measurementData[2] += 4
    # measurement

    #Repeating the above steps for a defined number of iterations
    #Step 4
    for _ in range(no_of_iterations): 
        centroids = []
        for idx in range(k):
            #Updating Centroids by taking mean of Cluster it belongs to
            temp_cent = x[points==idx].mean(axis=0) 
            centroids.append(temp_cent)
            # measurement
            measurementData[0] += 3
            measurementData[2] += 3
            # measurement
 
        centroids = np.vstack(centroids) #Updated Centroids 
         
        distances = cdist(x, centroids ,'euclidean')
        points = np.array([np.argmin(i) for i in distances])

        # measurement
        measurementData[0] += 5
        measurementData[2] += 5
        # measurement

    # measurement
    measurementData[0] += 1
    measurementData[2] += 1
    # measurement
    return points 
 
 
#Load Data
#data = load_digits().data#

# evaluate algorithm set for these values [10, 50, 100, 200, 500, 1000, 5000, 10000]
sizesOfSets = [10, 50, 100, 200, 500, 1000, 5000, 10000]

for size in sizesOfSets:
    center_1 = np.array([1,1])
    center_2 = np.array([5,5])
    center_3 = np.array([8,1])

    # Generate random data and center it to the three centers
    cluster_1 = np.random.randn(size, 2) + center_1
    cluster_2 = np.random.randn(size,2) + center_2
    cluster_3 = np.random.randn(size,2) + center_3

    data = np.concatenate((cluster_1, cluster_2, cluster_3), axis = 0)
    
    # declaring the list that will save assigns, comparisons, executed lines
    measurementData = [0, 0, 0]
    # Applying our function
    # measurement of execution time of desicion tree
    startTime = time.time()
    label = kmeans(data,3,size, measurementData)
    executionTime = (time.time() - startTime)
    # ##############################################
    print(f'\nsize of set: {size}')
    print(f'tiempo de ejecución en (s): {executionTime}')
    print(f'assigns: {measurementData[0]} / comparisons: {measurementData[1]} / executed lines of code: {measurementData[2]}\n')
 
# #Visualize the results
 
# u_labels = np.unique(label)
# for i in u_labels:
#     plt.scatter(data[label == i , 0] , data[label == i , 1] , label = i)
# plt.legend()
# plt.show()

