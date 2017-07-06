
#Testing tensorflow CNN 
'''
input -> convolve -> pooling (max_pooling) [which is one hidden layer usually 2 or more hidden layers ]
at the end fully connected layer [ which is also a hidden layer] -> output

'''
def get_training_list_from_images(dir_training):
    
    import os
    import tensorflow as tf
    list_dir_training = os.listdir(dir_training)
    num_classes = len(list_dir_training)
    print('Number of classes found: ' +  str(num_classes) + '\n')

    list_train_images = []
    list_train_labels = []

    for class_number in range(num_classes):
        dir_class = dir_training + '/' + list_dir_training[class_number]
        
        print dir_class

        list_dir_class = os.listdir(dir_class)
        num_images     = len(list_dir_class)

        for j in range(num_images):
            list_train_images.append(dir_class + '/'+ list_dir_class[j])
            list_train_labels.append(float(class_number + 1))

    return(list_train_images, list_train_labels)


#%%

import tensorflow as tf
import random
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes


#%%
n_classes = 10
batch_size = 128
keep_rate = 0.8


x = tf.placeholder('float', [28,28,3])
y = tf.placeholder('float')

keep_prob = tf.placeholder(tf.float32)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


def convolutional_neural_network(x):
    weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,3,32])),
               'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
               'W_fc':tf.Variable(tf.random_normal([7*7*64,1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, 28, 28, 3])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)
    
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2,[-1, 7*7*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out'])+biases['out']

    return output

def train_neural_network(x):
    from scipy import misc
    import numpy as np
    
    dir_training = '/home/oplab/sources/tensorflow/data_crust'
    list_train_images, list_train_labels = get_training_list_from_images(dir_training)


    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 3

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for i in range(int(30)):
                #epoch_x, epoch_y = mnist.train.next_batch(batch_size)

                img = misc.imread(list_train_images[i]) # 640x480x3 array    
                mat_image = np.array(img)
                mat_image = mat_image[0:28,0:28,:]


                _, c = sess.run([optimizer, cost], feed_dict={x: mat_image, y: list_train_labels[i]})
                print('a')
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        #print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
        
        coord.request_stop()
        coord.join(threads)
        sess.close()


train_neural_network(x)

    