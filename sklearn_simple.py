import sys
import os
import time
import numpy as np
from sklearn import metrics
import _compat_pickle as pickle

def navive_bayes_classifier(train_x,train_y):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.1)
    model.fit(train_x,train_y)
    return model
def logistic_regression_classifier(train_x,train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='12')
    model.fit(train_x,train_y)
    return model
def random_forest_classifier(train_x,train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=8)
    model.fit(train_x,train_y)
    return  model
def svm_classifier(train_x,train_y):
    from sklearn.svm import SVC
    model = SVC(kernel='rbf',probability=True)
    model.fit(train_x,train_y)
    return model
def read_data(data_file):
    import gzip
    f = gzip.open(data_file,'rb')
    train,val,test =pickle.load(f)
    f.close()
    train_x = train[0]
    train_y = train[1]
    test_x = test[0]
    test_y = test[1]
    return train_x,train_y,test_x,test_y
if __name__ == '__main__':
    data_file = "/tmp/data/mnist.pkl.gz"
    thresh = 0.5
    model_save_file = None
    model_save ={}
    test_classifiers = ['NB','LR','RF','SVM']
    classifiers =  {'NB':navive_bayes_classifier,
                    'LR':logistic_regression_classifier,
                    'RF':random_forest_classifier,
                    'SVM':svm_classifier
                    }

    print("reading training adn testing data......")
    train_x,train_y,test_x,test_y = read_data(data_file)
    num_train,num_feat=train_x.shape
    num_test,num_feat =test_y.shape
    print("np.unicode(train_y)--",np.unicode(train_y))
    is_binary_class = (len(np.unicode(train_y))==2)
    print("is_binary_class======",is_binary_class)
    print("****************** Data Info*************")
    print("#train data %d,#test Data: %d,dimension: %d",num_train,num_test,num_feat)
    for classifier in test_classifiers:
        print("&&&&&&&&&&&&&&&&&&&&%s&&&&&&&&&&&&&" %classifier)
        start_time = time.time()
        model =classifiers[classifier](train_x)(train_y)
        print("******train tokk %fs!",(time.time()-start_time))
        predict = model.predict(test_x)
        if model_save_file !=None:
            model_save[classifier] = model
        if is_binary_class:
            precision = metrics.precision_score((test_y,predict))
            recall = metrics.recall_score((test_y,predict))
            print("precision:%.2f%%,recall:%.2f%%" % (100*precision,100*recall))
        accuracy = metrics.accuracy_score((test_y,predict))
        print("accuracy %.2f%%" % (100*accuracy))
    if model_save_file !=None:
        pickle.dump(model_save,open(model_save_file),'wb')

