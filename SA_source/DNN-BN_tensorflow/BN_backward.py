import tensorflow as tf
import numpy as np
import BN_forward
import os
import h5py
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt

Modelpath = 'D:\\sa\\model\\BN-20epochs'
Modelname = 'ModelBN_augment'
logpath = 'D:\\sa\\model\\BN-20epochs\\log'

def create_placeholder(n_x, n_y):
    '''
    create a placeholder for tf session
    :param n_x: size of input sigmal, 3000 for 1 channel and 6000 for 2 combined channels.
    :param n_y: number of the signal-fragments
    :return: X,Y
    '''
    with tf.name_scope('inputs'):
        keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
        X = tf.placeholder(tf.float32, [None,n_x], name = 'Input_X')
        Y = tf.placeholder(tf.float32, [None,n_y], name = 'Input_Y')
        is_training = tf.placeholder(tf.bool, name='Mode')

    return X, Y, keep_prob, is_training

def one_hot_matrix(labels, C):

    C = tf.constant(C, name='C')
    one_hot_matrix = tf.one_hot(indices=labels, depth=C, axis=0)
    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()
    return one_hot


def random_mini_batch(X,Y,size):
    m = X.shape[0]
    mini_batches = []
    permutation = list(np.random.permutation(m))
    num = m // size
    shuffled_X = X[permutation,:]
    shuffled_Y = Y[permutation,:]
    for i in range(num):
        mini_batch_X = shuffled_X[i*size:(i+1)*size,:]
        mini_batch_Y = shuffled_Y[i*size:(i+1)*size,:,]
        mini_batch = (mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % size != 0:
        mini_batch_X = shuffled_X[num*size:,:]
        mini_batch_Y = shuffled_Y[num*size:,:]
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
    with tf.name_scope('loss'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = ZL, labels = Y))
        tf.summary.scalar('cost', cost)
    return cost

def L_model(X_train, Y_train, X_test, Y_test,  learning_rate = 0.0001, num_epoch = 2000, mini_batch_size = 64, kb = 0.5, print_cost = True, is_train = False):
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
    (m, n_x) = X_train.shape
    num_mini = int(m / mini_batch_size)
    print(num_mini)
    n_y = Y_train.shape[1]
    costs = []
    X, Y, keep_prob, is_training = create_placeholder(n_x, n_y)
    parameters = BN_forward.initialize_para()
    global_step = tf.Variable(0, trainable= False)
    ZL = BN_forward.forward_prob(X, parameters, keep_prob, is_training)
    cost = compute_cost(ZL, Y)
    #learning_rate = tf.train.exponential_decay(learning_rate_base, global_step, 1250, 0.99, staircase=True)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost, global_step)
    init = tf.global_variables_initializer()
    #var_list = tf.trainable_variables()
    #g_list = tf.global_variables()
    #bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    #bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    #var_list += bn_moving_vars
    #saver = tf.train.Saver(var_list=var_list, max_to_keep=5)
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
                _, mini_batch_cost, step = sess.run([optimizer, cost, global_step],
                                                    feed_dict = {X: mini_X, Y: mini_Y, keep_prob: kb, is_training: is_train})
                epoch_cost += mini_batch_cost/ num_mini
            if print_cost == True and epoch % 50 == 0:
                print('cost after epoch ', (epoch, epoch_cost))
                costs.append(epoch_cost)
                result = sess.run(merged, feed_dict={X: mini_X, Y: mini_Y, keep_prob: kb, is_training: is_train})
                writer.add_summary(result, epoch)
            if print_cost == True and epoch % 50 == 0:
                saver.save(sess, os.path.join(Modelpath, Modelname), global_step = step)
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('Iteration times')
        plt.title('learning_rate  = %s' % learning_rate)
        plt.show()
        parameters = sess.run(parameters)
        #tf.argmax: return the subscripts of the max number per column
        results = sess.run(tf.transpose(tf.argmax(ZL, axis = 1)), feed_dict={X: X_test, Y: Y_test, keep_prob: 1, is_training: False})
        correct_prediction = tf.equal(tf.argmax(ZL, axis = 1), tf.argmax(Y, axis = 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print('Train accuracy:', accuracy.eval({X: X_train, Y: Y_train, keep_prob: 1, is_training: False}))
        print('Test accuracy', accuracy.eval({X: X_test, Y: Y_test, keep_prob: 1, is_training: False}))
        return parameters, results

def data_augmentation(X_train, EEG_Add_path, mean, std):
    file = h5py.File(EEG_Add_path, 'r')
    EEG_1_Add = file['EEG_1'][:]
    EEG_2_Add = file['EEG_2'][:]
    file.close()
    EEG_Add = np.concatenate((EEG_1_Add, EEG_2_Add), axis=0)
    EEG_Add -= mean
    EEG_Add /= std
    X_train = np.concatenate((X_train, EEG_Add.T), axis = 0)
    return X_train

def labels_augmentation(Y_train, Labels_Add_path):
    file = h5py.File(Labels_Add_path, 'r')
    Labels_Add = file['train_set_num'][:]
    file.close()
    Labels_Add = np.squeeze(Labels_Add)
    Labels__Add_one = one_hot_matrix(Labels_Add, 6)
    Y_train = np.concatenate((Y_train, Labels__Add_one.T), axis = 0)
    return Y_train





def main():
    filepath = 'D:\\sa\\features\\TrainSet_20186501.h5'
    file = h5py.File(filepath, 'r')
    EEG_1 = file['EEG_1'][:]
    EEG_2 = file['EEG_2'][:]
    file.close()
    EEG_Add_path = 'D:\\sa\\features\\features_add.h5'
    EEG = np.concatenate((EEG_1, EEG_2), axis=0)
    mean = np.reshape(np.mean(EEG, axis = 1), [EEG.shape[0],1])
    std = np.reshape(np.std(EEG, axis = 1), [EEG.shape[0],1])
    EEG -= mean
    EEG /= std
    print(np.mean(EEG[0,:]))
    print(np.std(EEG[0,:]))
    #X_train = EEG[:, 0:80000].T   #for 2 channels
    X_train = EEG[:, 0:80000].T #for 2 channel
    #X_train = data_augmentation(X_train, EEG_Add_path, mean, std)
    #X_dev = EEG[:, 80000:90000].T
    X_dev = EEG[:, 80000:].T

    labelpath = 'D:\\sa\\features\\TrainSet_20186500.h5'
    file = h5py.File(labelpath, 'r')
    labels = file['train_set_num'][:]
    file.close()
    labels = np.squeeze(labels)
    labels_one = one_hot_matrix(labels, 6)
    Y_train = labels_one[:, 0:80000].T
    #Labels_Add_path = 'D:\\sa\\features\\data_add.h5'
    #Y_train = labels_augmentation(Y_train, Labels_Add_path)
    Y_dev = labels_one[:, 80000:].T

    parameters, results = L_model(X_train, Y_train, X_dev, Y_dev, learning_rate=0.0001,
                                  num_epoch=101, mini_batch_size=64, kb=0.5, print_cost=True, is_train = True)

if __name__ == "__main__":
   main()

