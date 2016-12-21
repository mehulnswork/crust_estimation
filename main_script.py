# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 11:34:49 2016
Predicting the split image properties for Cruse estimation 
@author: Sangekar Mehul

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

def get_split_imagelist(dir_splits):
    import os
    d = os.listdir(dir_splits)
    split_list = [ s for s in d if '.jpeg' in s]
    print('Number of JPEG files found:' + str(len(split_list)))
    return split_list    

def get_image_stats(dir_splits, split_list):
    path_image = dir_splits + '/' + split_list[0]
    cv2_img, img_height, img_width =  load_opencv_image(path_image)
    avg_r, avg_g, avg_b = get_avg_rgb(cv2_img)
    avg_h, avg_s, avg_v = get_avg_hsv(cv2_img)
    return None

def load_opencv_image(path_image):
    import cv2
    cv2_image  = cv2.imread(path_image)
    img_width, img_height, channels = cv2_image.shape
    return(cv2_image, img_height, img_width)
    
def get_avg_rgb(image):
    import numpy as np
    avg_row   = np.average(image, axis=0)    
    avg_color = np.average(avg_row, axis=0)
    avg_r = avg_color[0]
    avg_g = avg_color[1]
    avg_b = avg_color[2]    
    return(avg_r, avg_g, avg_b)    

def get_avg_hsv(image):
    import numpy as np
    import cv2
    hsv       = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    avg_row   = np.average(hsv, axis=0)    
    avg_hsv   = np.average(avg_row, axis=0)
    avg_h = avg_hsv[0]
    avg_s = avg_hsv[1]
    avg_v = avg_hsv[2]    
    return(avg_h, avg_s, avg_v)       

#%% Reading blocks from folder


dir_splits = '/home/oplab/cruiseData/NT15-03/HPD1780/Publish/Splits'
split_list = get_split_imagelist(dir_splits)
get_image_stats(dir_splits, split_list)

import numpy as np
from sklearn import datasets



iris = datasets.load_iris()
iris_X = iris.data
iris_Y = iris.target
np.random.seed(0)
indices = np.random.permutation(len(iris_X))
para_train_list   = iris_X[indices[:-10]] #all except last 10
labels_train_list = iris_Y[indices[:-10]]
para_predict_list  = iris_X[indices[-10:]] # only last 10
labels_verify_list  = iris_Y[indices[-10:]]

print('Correct labels:' + str(labels_verify_list))

svc = create_svm()
train_crust_images(para_train_list, labels_train_list, svc)
labels_predict_list = predict_crust_images(para_predict_list, svc)

print('SCM labels:' + str(labels_predict_list))
print('Program exit')

#%% Extracting parameters from each block for analysis



#%% Unsupervised clustering of blocks into groups








