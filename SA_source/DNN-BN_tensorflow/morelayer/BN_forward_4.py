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
        W1 = tf.get_variable("W1", [64, 80], initializer=tf.contrib.layers.xavier_initializer())
        #W1 = tf.get_variable("W1", [32, 80], initializer=tf.contrib.layers.xavier_initializer())
        tf.summary.histogram('layer1/weights', W1)

    with tf.name_scope('layer2'):
        W2 = tf.get_variable("W2", [80, 80], initializer=tf.contrib.layers.xavier_initializer())
        tf.summary.histogram('layer2/weights', W2)

    with tf.name_scope('layer3'):
        W3 = tf.get_variable("W3", [80, 80], initializer=tf.contrib.layers.xavier_initializer())
        tf.summary.histogram('layer3/weights', W3)

    with tf.name_scope('layer5'):
        W4 = tf.get_variable("W4", [80, 6], initializer=tf.contrib.layers.xavier_initializer())
        tf.summary.histogram('layer4/weights', W4)
        b4 = tf.get_variable("b4", [1, 6], initializer=tf.zeros_initializer())
        tf.summary.histogram('layer4/weights', b4)
    parameters = {"W1": W1,
                  "W2": W2,
                  "W3": W3,
                  "W4": W4,
                  "b4": b4}

    return parameters

def batch_norm_wrapper(inputs,  is_training, epsilon = 0.001):
    '''
    fc_mean, fc_var = tf.nn.moments(inputs,axes=[1])
    scale = tf.Variable(tf.ones([size]))
    shift = tf.Variable(tf.zeros([size]))
    ema = tf.train.ExponentialMovingAverage(decay=0.5)

    def mean_var_with_update():
        ema_apply_op = ema.apply([fc_mean, fc_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(fc_mean), tf.identity(fc_var)

    mean, var = tf.cond(is_training,
                        mean_var_with_update(),
                        lambda:(
                            ema.average(fc_mean),ema.average(fc_var)))


    bn = tf.nn.batch_normalization(inputs, mean, var, shift, scale, epsilon = epsilon)
    '''
    scale = tf.Variable(tf.ones([inputs.get_shape()[0],1]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[0],1]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[0],1]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[0],1]), trainable=False)

    decay = 0.999
    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs,axes = [1])
        train_mean = tf.assign(pop_mean , pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)






def forward_prob(X, parameters, keep_prob,  is_training = False):
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
    #b1 = parameters['b1']
    W2 = parameters['W2']
    #b2 = parameters['b2']
    W3 = parameters['W3']

    W4 = parameters['W4']
    b4 = parameters['b4']


    Z1 = tf.matmul(X,W1)  # Z1 = np.dot(W1, X) + b1
    #bn1 = batch_norm_wrapper(Z1,  is_training = is_training)
    bn1 = tf.layers.batch_normalization(Z1,  training = is_training)
    Z1 = tf.nn.dropout(bn1, keep_prob)
    A1 = tf.nn.relu(Z1)  # A1 = relu(Z1)

    Z2 = tf.matmul(A1, W2)  # Z2 = np.dot(W2, a1) + b2
    #bn2 = batch_norm_wrapper(Z2,  is_training=is_training)
    bn2 = tf.layers.batch_normalization(Z2, axis = 1, training= is_training)
    Z2 = tf.nn.dropout(bn2, keep_prob)
    A2 = tf.nn.relu(Z2)  # A2 = relu(Z2)

    Z3 = tf.matmul(A2, W3)  # Z2 = np.dot(W2, a1) + b2
    #bn2 = batch_norm_wrapper(Z2,  is_training=is_training)
    bn3 = tf.layers.batch_normalization(Z3, axis = 1, training= is_training)
    Z3 = tf.nn.dropout(bn3, keep_prob)
    A3 = tf.nn.relu(Z3)  # A2 = relu(Z2)

    ZL = tf.add(tf.matmul(A3, W4), b4)  # Z3 = np.dot(W3,Z2) + b3
    tf.summary.histogram('output',ZL)

    return ZL