# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 18:08:46 2016

@author: oplab
"""



def predict_crust_images(para_list, svc_class):
    predict_list =  svc_class.predict(para_list)        
    return predict_list
    
def train_crust_images(para_list, labels_list, svc_class):
    svc_class.fit(para_list, labels_list) 
    return None

def create_svm():
    from sklearn import svm
    svc_class = svm.SVC(kernel='poly')        
    return svc_class
    
def create_iris_dataset():
    from sklearn import datasets    
    import numpy as np
    iris = datasets.load_iris()
    iris_X = iris.data
    iris_Y = iris.target
    np.random.seed(0)
    indices = np.random.permutation(len(iris_X))
    para_train_list   = iris_X[indices[:-10]] #all except last 10
    labels_train_list = iris_Y[indices[:-10]]
    para_predict_list  = iris_X[indices[-10:]] # only last 10
    labels_verify_list  = iris_Y[indices[-10:]]    
    return(para_train_list, labels_train_list, para_predict_list, labels_verify_list )
    
    

