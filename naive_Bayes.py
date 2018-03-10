
def loadDataset():
    postingList = [[
        'asd','asd','asdfs','asdasd'],
        ['asda','dsada','asdads','asdas']
        ['qwe','qweq','qweq' ]]
    classVec = [0,1,1,1,1]
    return postingList,classVec

def createVocabList(dataSet):
    vocabset = set([])
    for doucment in dataset:
        vocabset = vocabset| set(document)
    print("2=====",list(vocabset))
    return list(vocabset)

def setofWord2Vec(vocabList,inputsSet):
    returnVec = [0]*len(vocabList)
    for word in inputsSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] =1
        else:
            print("4=======the word:%s ot in my Vocabbulary" % word)
    print("5========",returnVec)
    return returnVec
def trainNB0(trainMtrix,trainCategory):
    numTrainDoc = len(trainMatrix)
    numWords = len(Matraix[0])
    p0Num =ones(numWords)
    p1Num =ones(nemWords)
    p0Dem = 2.0
    p1Dem =3.0 #初始化参数
    for i in range(numtrainDoc):
        if trainCategory[i] ==1:
            p1Num +=trainMtrix[i]
            p1Dem +=sum(trainMtrix)
        else:
            p0Num += trainMtrix[i]
            p0Dem += sum(trainMtrix)
    p1Vec =p1Num/p1Dem
    return p1Vec

if __name__=="__main__":
    testingNB()

def testingNb():
    list0Post,listClasses = loadDataset()#产生数据
    myvocabList = createVocalList(list0Post)
    print("1======",myvocabList)
    trainMat = [] #创建一个空集
    for postinDoc in list0Post:
        trainMat.append(setofWord2Vec(myvocabList,postinDoc))
    print("3======",array(trainMat))
    p0V,p1V,pAv = trainNB0(array(trainMat),array(listClasses))#训练数据
    