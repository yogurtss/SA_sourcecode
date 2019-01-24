import tensorflow as tf
import numpy as np
import forward
import os
import h5py
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt

Modelpath = 'D:\\sa\\model\\3'
Modelname = 'Model2'
logpath = 'D:\\sa\\model\\3\\log'

def create_placeholder(n_x, n_y):
    '''
    create a placeholder for tf session
    :param n_x: size of input sigmal, 3000 for 1 channel and 6000 for 2 combined channels.
    :param n_y: number of the signal-fragments
    :return: X,Y
    '''
    with tf.name_scope('inputs'):
        keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
        X = tf.placeholder(tf.float32, [n_x, None], name = 'Input_X')
        Y = tf.placeholder(tf.float32, [n_y, None], name = 'Input_Y')

    return X, Y, keep_prob

def one_hot_matrix(labels, C):

    C = tf.constant(C, name='C')
    one_hot_matrix = tf.one_hot(indices=labels, depth=C, axis=0)
    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()
    return one_hot


def random_mini_batch(X,Y,size):
    m = X.shape[1]
    mini_batches = []
    permutation = list(np.random.permutation(m))
    num = m // size
    shuffled_X = X[:,permutation]
    shuffled_Y = Y[:,permutation]
    for i in range(num):
        mini_batch_X = shuffled_X[:,i*size:(i+1)*size]
        mini_batch_Y = shuffled_Y[:,i*size:(i+1)*size]
        mini_batch = (mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % size != 0:
        mini_batch_X = shuffled_X[:,num*size:]
        mini_batch_Y = shuffled_Y[:,num*size:]
        mini_batch = (mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches

def compute_cost(ZL, Y):
    '''
    compute the cost
    :param ZL: output of forward propagation
    :param Y: labels(true ouptput)
    :return:
    '''
    logits = tf.transpose(ZL)
    labels = tf.transpose(Y)
    with tf.name_scope('loss'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
        tf.summary.scalar('cost', cost)
    return cost

def L_model(X_train, Y_train, X_test, Y_test,  learning_rate = 0.0001, num_epoch = 2000, mini_batch_size = 64, kb = 0.5, print_cost = True):
    '''
    Implements a L-layers neural network with tensorflow
    :param X_train: training-set, dtype should be float32, size(eg. (3000, 10000)or (6000, 10000)
    :param Y_train: training-labels. size(eg. (6,10000))
    :param X_test: test-set
    :param Y_test:
    :param layers_dims: structure of NN
    :param learning_rate:
    :param num_epoch: number of iterations times
    :param mini_batch_size:
    :param print_cost:
    :return:
    '''
    ops.reset_default_graph()
    (n_x, m) = X_train.shape
    num_mini = int(m / mini_batch_size)
    print(num_mini)
    n_y = Y_train.shape[0]
    costs = []
    X, Y, keep_prob = create_placeholder(n_x, n_y)
    parameters = forward.initialize_para()
    global_step = tf.Variable(0, trainable= False)
    ZL = forward.forward_prob(X, parameters, keep_prob)
    cost = compute_cost(ZL, Y)
    #learning_rate = tf.train.exponential_decay(learning_rate_base, global_step, 1250, 0.99, staircase=True)
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost, global_step)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(logpath, sess.graph)
        sess.run(init)
        ckpt = tf.train.get_checkpoint_state(Modelpath)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        for epoch in range(num_epoch):
            epoch_cost = 0.
            minibatches = random_mini_batch(X_train, Y_train, mini_batch_size)
            for minibatch in minibatches:
                (mini_X, mini_Y) = minibatch
                _, mini_batch_cost, step = sess.run([optimizer, cost, global_step], feed_dict = {X: mini_X, Y: mini_Y, keep_prob: kb})
                epoch_cost += mini_batch_cost/ num_mini
            if print_cost == True and epoch % 50 == 0:
                print('cost after epoch ', (epoch, epoch_cost))
                costs.append(epoch_cost)
                result = sess.run(merged, feed_dict={X: mini_X, Y: mini_Y, keep_prob: kb})
                writer.add_summary(result, epoch)
            if print_cost == True and epoch % 1000 == 0:
                saver.save(sess, os.path.join(Modelpath, Modelname), global_step = step)
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('Iteration times')
        plt.title('learning_rate  = %s' % learning_rate)
        plt.show()
        parameters = sess.run(parameters)
        #tf.argmax: return the subscripts of the max number per column
        results = sess.run(tf.argmax(ZL), feed_dict={X: X_test, Y: Y_test, keep_prob: 1})
        correct_prediction = tf.equal(tf.argmax(ZL), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print('Train accuracy:', accuracy.eval({X: X_train, Y: Y_train, keep_prob: 1}))
        print('Test accuracy', accuracy.eval({X: X_test, Y: Y_test, keep_prob: 1}))
        return parameters, results

def main():
    filepath = 'D:\\sa\\features\\TrainSet_20186501.h5'
    file = h5py.File(filepath, 'r')
    EEG_1 = file['EEG_1'][:]
    EEG_2 = file['EEG_2'][:]
    EEG = np.concatenate((EEG_1, EEG_2), axis=0)
    X_train = EEG[:, 0:80000]
    X_dev = EEG[:, 80000:90000]
    file.close()

    labelpath = 'D:\\sa\\features\\TrainSet_20186500.h5'
    file = h5py.File(labelpath, 'r')
    labels = file['train_set_num'][:]
    labels = np.squeeze(labels)
    labels_one = one_hot_matrix(labels, 6)
    Y_train = labels_one[:, 0:80000]
    Y_dev = labels_one[:, 80000:90000]
    file.close()
    parameters, results = L_model(X_train, Y_train, X_dev, Y_dev, learning_rate_base=0.0001,
                                  num_epoch=3001, mini_batch_size=64, kb=0.5, print_cost=True)

if __name__ == "__main__":
   main()

