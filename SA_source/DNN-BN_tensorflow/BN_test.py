import tensorflow as tf
import numpy as np
import h5py
import os
from tensorflow.python.framework import ops
import BN_forward
import BN_backward
n_x = 64
n_y = 6

def sleep_stage_test(X_test, Y_test):
    with tf.Graph().as_default() as g:
        X = tf.placeholder(tf.float32, [None, n_x])
        Y = tf.placeholder(tf.float32, [None, n_y])
        parameters = BN_forward.initialize_para()
        Y_prediction = BN_forward.forward_prob(X, parameters, 1, is_training = False)
        saver = tf.train.Saver()
        correct_prediction = tf.equal(tf.argmax(Y_prediction, axis = 1), tf.argmax(Y, axis = 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(BN_backward.Modelpath)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                accuracy_store = sess.run(accuracy, feed_dict= {X: X_test, Y: Y_test})
                print('after %s training-steps, dev/test accuracy = %g' % (global_step, accuracy_store))
            else:
                print('No Model file found')

def main():
    filepath = 'D:\\sa\\features\\TrainSet_20186501.h5'
    file = h5py.File(filepath, 'r')
    EEG_1 = file['EEG_1'][:]
    EEG_2 = file['EEG_2'][:]
    EEG = np.concatenate((EEG_1, EEG_2), axis=0)
    mean = np.reshape(np.mean(EEG, axis=1), [EEG.shape[0], 1])
    std = np.reshape(np.std(EEG, axis=1), [EEG.shape[0], 1])
    EEG -= mean
    EEG /= std
    print(np.mean(EEG[0, :]))
    print(np.std(EEG[0, :]))
    X_test = EEG[:, 90000:100000].T
    file.close()

    labelpath = 'D:\\sa\\features\\TrainSet_20186500.h5'
    file = h5py.File(labelpath, 'r')
    labels = file['train_set_num'][:]
    labels = np.squeeze(labels)
    labels_one = BN_backward.one_hot_matrix(labels, 6)
    Y_test= labels_one[:,90000:100000].T
    sleep_stage_test(X_test, Y_test)

if __name__ == "__main__":
    main()
    print('test')
