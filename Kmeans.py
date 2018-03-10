
from numpy import *
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def Kmeans (dataSet,k):
    sampleNum,col = dataSet.shape
    print(dataSet.shape)
    cluster = mat(((sampleNum,2)))
    print("cluster----------",cluster)
    centroids = zeros((k,col))
    print("centrois----------",centroids)
    for i in range(k):
        index = int(random.uniform(0,sampleNum))
        print("index------",index)
        centroids[i,:] = dataSet[index,:]
        print("centroids[i,:]",centroids[i,:])
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(sampleNum):
            minDist = sqrt(sum(power(centroids[0,:]-dataSet[i,:],2)))
            minIndex = 0
            for j in range(1,k):
                distance = sqrt(sum(power(centroids[j,:]-dataSet[i,:],2)))
                if distance < minDist:
                    minIndex = j
                    minDist = distance
            if  cluster[i, 0] != minIndex:
                clusterChanged = True
                cluster[i,:] = minIndex,minDist**2
        for j in range(k):
            pointsIncluster = dataSet[nonzero(cluster[:,0].A==j)[0]]
            centroids[j,:]= mean(pointsIncluster,axis=0)
            print("pointsIncluster", pointsIncluster)
    print("return1",centroids)
    print("return2", cluster)
    return centroids,cluster

dataSet = [[1,1],[3,1],[1,4],[2,5],[11,12],[14,11],[13,12],[11,16],[17,12],[28,10],[26,15],[27,13],[28,11]
           ,[29,15
    ]]
dataSet = mat(dataSet)
print("dataSet",dataSet)
k =3
centroids,cluster = Kmeans(dataSet,k)
sampleNum,col = dataSet.shape
mark = ['or','ob','og']

for i in range(sampleNum):
    print("cluster[i,0]",cluster[i,0])
    markindex =  int(cluster[i,0])
    plt.plot(dataSet[i,0],dataSet[i,1],mark[markindex])
mark = ['+r','+b','+b']
for i in range(k):
    plt.plot(centroids[i,0],centroids[i,1],mark[i],markersize=12)
plt.show()
