from numpy import *
filename='....\\testSet.txt'
def loadDateSet():
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat
def sigmoid(inX):
    return 1.0/(1+exp(-inX))
def gradAscent(dataMat,labelMat):
    dataMatrix=mat(dataMat)#将读取数据转换为矩阵
    classLabers=mat(labelMat).transpose()
    m,n=shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weight = ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weight)
        error = classLabers - h
        weights = weights -alpha*dataMatrix.transpose()*error
    return weights
