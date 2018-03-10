
from numpy import *
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

dataSet = [[1,1],[3,1],[1,4],[2,5],[11,12],[14,11],[13,12],[11,16],[17,12],[28,10],[26,15],[27,13],[28,11]
           ,[29,15]]
dataSet= mat(dataSet)
k=3
markers = ['^','o','x']
cls = KMeans(k).fit(dataSet)
for i in range(k):
    members = cls.labels_==i
    plt.scatter(dataSet[members,0].tolist(),dataSet[members,1].tolist(),marker=markers[i])
plt.show()