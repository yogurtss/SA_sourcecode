import tensorflow as tf
import numpy as np
import h5py
import os
from tensorflow.python.framework import ops
import forward
import backward
n_x = 64
n_y = 6

def sleep_stage_test(X_test, Y_test):
    with tf.Graph().as_default() as g:
        X = tf.placeholder(tf.float32, [n_x, None])
        Y = tf.placeholder(tf.float32, [n_y, None])
        parameters = forward.initialize_para()
        Y_prediction = forward.forward_prob(X, parameters, 1)
        saver = tf.train.Saver()
        correct_prediction = tf.equal(tf.argmax(Y_prediction), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(backward.Modelpath)
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
    EEG = np.concatenate((EEG_1, EEG_2), axis = 0)
    X_test = EEG[:,80000:90000]
    file.close()

    labelpath = 'D:\\sa\\features\\TrainSet_20186500.h5'
    file = h5py.File(labelpath, 'r')
    labels = file['train_set_num'][:]
    labels = np.squeeze(labels)
    labels_one = backward.one_hot_matrix(labels, 6)
    Y_test= labels_one[:,80000:90000]
    sleep_stage_test(X_test, Y_test)

if __name__ == "__main__":
    main()

