import tensorflow as tf
import numpy as np

def initialize_para():
    '''
    Initialize parameters to build network with tf.
    layers_dims: number of nodes for each layer
    :return:
    '''
    parameters = {}
    '''
    for i in range(1,L):
        # He Initialization

        parameters['W' + str(i)] = tf.get_variable('W' + str(i), [layers_dims[i], layers_dims[i-1]], initializer=tf.contrib.layers.xavier_initializer())
        parameters['b' + str(i)] = tf.get_variable('b' + str(i), [layers_dims[i], 1], initializer = tf.zeros_initializer())

        #He Initialization
        #parameters['W' + str(i)] = tf.Variable(np.random.randn(layers_dims[i], layers_dims[i - 1]) * np.sqrt(2 / layers_dims[i - 1]),name = 'W' + str(i), dtype = tf.float32)
        #parameters['b' + str(i)] = tf.get_variable('b' + str(i), [layers_dims[i], 1],initializer=tf.zeros_initializer())
    '''
    '''
    with tf.name_scope('layer1'):
        W1 = tf.Variable(np.random.randn(80, 64), dtype=tf.float32) * np.sqrt(2 / 64)
        tf.summary.histogram('layer1/weights', W1)
        b1 = tf.get_variable("b1", [80, 1], initializer=tf.zeros_initializer())
        tf.summary.histogram('layer1/biases', b1)
    with tf.name_scope('layer2'):
        W2 = tf.Variable(np.random.randn(80, 80), dtype=tf.float32) * np.sqrt(2 / 80)
        tf.summary.histogram('layer1/weights', W2)
        b2 = tf.get_variable("b2", [80, 1], initializer=tf.zeros_initializer())
        tf.summary.histogram('layer1/weights', b2)
    with tf.name_scope('layer3'):
        W3 = tf.Variable(np.random.randn(6, 80), dtype=tf.float32) * np.sqrt(2 / 80)
        tf.summary.histogram('layer1/weights', W3)
        b3 = tf.get_variable("b3", [6, 1], initializer=tf.zeros_initializer())
        tf.summary.histogram('layer1/weights', b3)
        '''
    with tf.name_scope('layer1'):
        W1 = tf.get_variable("W1", [80, 64], initializer=tf.contrib.layers.xavier_initializer())
        tf.summary.histogram('layer1/weights', W1)
        b1 = tf.get_variable("b1", [80, 1], initializer=tf.zeros_initializer())
        tf.summary.histogram('layer1/biases', b1)
    with tf.name_scope('layer2'):
        W2 = tf.get_variable("W2", [80, 80], initializer=tf.contrib.layers.xavier_initializer())
        tf.summary.histogram('layer1/weights', W2)
        b2 = tf.get_variable("b2", [80, 1], initializer=tf.zeros_initializer())
        tf.summary.histogram('layer1/weights', b2)
    with tf.name_scope('layer3'):
        W3 = tf.get_variable("W3", [6, 80], initializer=tf.contrib.layers.xavier_initializer())
        tf.summary.histogram('layer1/weights', W3)
        b3 = tf.get_variable("b3", [6, 1], initializer=tf.zeros_initializer())
        tf.summary.histogram('layer1/weights', b3)
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters

def forward_prob(X, parameters, keep_prob):
    '''
    A = X
    L = len(parameters)//2
    for i in range(1,L):
        A_prev = A
        Z = tf.add(tf.matmul(parameters['W' + str(i)], A_prev), parameters['b' + str(i)])
        A = tf.nn.relu(Z)
    ZL = tf.add(tf.matmul(parameters['W' + str(L)], A), parameters['b' + str(L)])
    '''
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']


    Z1 = tf.add(tf.matmul(W1, X), b1)  # Z1 = np.dot(W1, X) + b1
    Z1 = tf.nn.dropout(Z1, keep_prob)
    A1 = tf.nn.relu(Z1)  # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)  # Z2 = np.dot(W2, a1) + b2
    Z2 = tf.nn.dropout(Z2, keep_prob)
    A2 = tf.nn.relu(Z2)  # A2 = relu(Z2)
    ZL = tf.add(tf.matmul(W3, A2), b3)  # Z3 = np.dot(W3,Z2) + b3
    tf.summary.histogram('output',ZL)

    return ZL