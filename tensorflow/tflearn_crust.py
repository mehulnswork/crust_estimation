"""
Modified for the crust data 
Author: M
Based on the tflearn CIFAR-10 example at:
https://github.com/tflearn/tflearn/blob/master/examples/images/convnet_cifar10.py
"""

from __future__ import division, print_function, absolute_import

#from skimage import color
import numpy as np
import os

from skimage import io
from scipy.misc import imresize
from sklearn.cross_validation import train_test_split

import tflearn
from tflearn.data_utils import to_categorical
#from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.metrics import Accuracy


#%% get training images from folders

def get_training_list_from_images(dir_training):
    
    list_dir_training = os.listdir(dir_training)
    num_classes       = len(list_dir_training)
    
    print('Number of classes found: ' +  str(num_classes) + '\n')

    list_train_images = []
    list_train_labels = []

    for class_number in range(num_classes):
        dir_class = dir_training + '/' + list_dir_training[class_number]
        list_dir_class = os.listdir(dir_class)
        num_images     = len(list_dir_class)

        for j in range(num_images):
            list_train_images.append(dir_class + '/'+ list_dir_class[j])
            list_train_labels.append(int(class_number))

    return(list_train_images, list_train_labels, num_classes)


#%% get testing images from folders

def get_testing_list_from_images(dir_testing):
    
    list_dir_testing = os.listdir(dir_testing)
    num_images       = len(list_dir_testing)    

    list_testing_images = []

    for image_number in range(num_images):
        list_testing_images.append(dir_testing + '/' + list_dir_testing[image_number])

    return(list_testing_images)



dir_training = '/home/oplab/sources/tensorflow/data_crust/train'
list_train_images, list_train_labels, num_classes = get_training_list_from_images(dir_training)

n_files = len(list_train_labels)

image_size  = 136
image_depth = 3

allX = np.zeros((n_files, image_size, image_size, image_depth), dtype='float64')
ally = np.zeros(n_files)

count = 0

for f in range(n_files):
    try:
        img = io.imread(list_train_images[f])
        new_img = imresize(img, (image_size, image_size, image_depth))
        allX[count] = np.array(new_img)
        ally[count] = list_train_labels[f]
        count += 1
    except:
        continue
    
dir_testing = '/home/oplab/sources/tensorflow/data_crust/test'
list_testing_images = get_testing_list_from_images(dir_testing)

n_test_files = len(list_testing_images)

allX_test = np.zeros((n_test_files, image_size, image_size, image_depth), dtype='float64')

count = 0

for f in range(n_test_files):
    try:
        img = io.imread(list_testing_images[f])
        new_img = imresize(img, (image_size, image_size, image_depth))
        allX_test[count] = np.array(new_img)
        count += 1
    except:
        continue

# test-train split   
X, X_test, Y, Y_test = train_test_split(allX, ally, test_size=0.1, random_state=42)

# encode the Ys
Y = to_categorical(Y, num_classes)
Y_test = to_categorical(Y_test, num_classes)

###################################
# Image transformations
###################################

# normalisation of images
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Create extra synthetic training data by flipping & rotating images
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

###################################
# Define network architecture
###################################

# Input is a 32x32 image with 3 color channels (red, green and blue)
network = input_data(shape=[None, image_size, image_size, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

# 1: Convolution layer with 32 filters, each 3x3x3
conv_1 = conv_2d(network, 32, 3, activation='relu', name='conv_1')

# 2: Max pooling layer
network = max_pool_2d(conv_1, 2)

# 3: Convolution layer with 64 filters
conv_2 = conv_2d(network, 64, 3, activation='relu', name='conv_2')

# 4: Convolution layer with 64 filters
conv_3 = conv_2d(conv_2, 64, 3, activation='relu', name='conv_3')

# 5: Max pooling layer
network = max_pool_2d(conv_3, 2)

# 6: Fully-connected 512 node layer
network = fully_connected(network, 512, activation='relu')

# 7: Dropout layer to combat overfitting
network = dropout(network, 0.5)

# 8: Fully-connected layer with two outputs
network = fully_connected(network, num_classes, activation='softmax')

# Configure how the network will be trained
acc = Accuracy(name="Accuracy")
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.0005, metric=acc)

# Wrap the network in a model object
model = tflearn.DNN(network, checkpoint_path='tmp/model_cat_dog_6.tflearn', max_checkpoints = 3,
                    tensorboard_verbose = 3, tensorboard_dir='tmp/tflearn_logs/')


#%% Train model for N epochs

model.fit(X, Y, validation_set=(X_test, Y_test), batch_size=500,
      n_epoch=75, run_id='model_cat_dog_6', show_metric=True)

model.save('model_cat_dog_6_final.tflearn')

print('Predicting')

results = model.predict(allX_test)

dir_classify = '/home/oplab/sources/tensorflow/splits_classfied_kmeans.csv'
file_classify = open(dir_classify, 'w')

for i in range(len(allX_test)):
    res      = (np.argmax(results[i]))
    img_name = list_testing_images[i]
    file_classify.write(img_name + ',' + str(res) + '\n')
    
file_classify.close()
print('Completed')