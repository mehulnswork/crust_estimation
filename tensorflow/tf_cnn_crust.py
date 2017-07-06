
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
            list_train_labels.append(class_number + 1)

    return(list_train_images, list_train_labels)


#%%

import tensorflow as tf
import random
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

dir_training = '/home/oplab/sources/tensorflow/data_crust'
list_train_images, list_train_labels = get_training_list_from_images(dir_training)

#%%

train_images = ops.convert_to_tensor(list_train_images, dtype=dtypes.string)
train_labels = ops.convert_to_tensor(list_train_labels, dtype=dtypes.int32)

train_input_queue = tf.train.slice_input_producer([train_images, train_labels], shuffle=False)

file_content = tf.read_file(train_input_queue[0])
train_image = tf.image.decode_jpeg(file_content, channels=3)
train_image = tf.image.resize_images(train_image, [137,137])
train_label = train_input_queue[1]
train_image.set_shape([137, 137, 3])


#train_image_batch, train_label_batch = tf.train.batch([train_image, train_label], batch_size=10 )

with tf.Session() as sess:
  
  # initialize the variables
  sess.run(tf.global_variables_initializer())
  
  # initialize the queue threads to start to shovel data
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  print "from the train set:"
  for i in range(20):
    print sess.run(train_label)

  # stop our queue threads and properly close the session
  coord.request_stop()
  coord.join(threads)
  sess.close()
    