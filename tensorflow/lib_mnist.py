
#Testing tensorflow CNN 
'''
input -> convolve -> pooling (max_pooling) [which is one hidden layer usually 2 or more hidden layers ]
at the end fully connected layer [ which is also a hidden layer] -> output

'''

def load_data(dir_mnist):
    import tensorflow as tf
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets(dir_mnist, one_hot = True)
    return None
