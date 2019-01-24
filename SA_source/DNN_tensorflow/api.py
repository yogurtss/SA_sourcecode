import tensorflow as tf
import numpy as np
import h5py
import forward
import backward
import test
import tools

def restore_model(X_test):
    '''

    :param X: size must be [64*n], here n is number of samples you want to prediction.
    :return: sleep stage
    '''
    with tf.Graph().as_default() as g:
        X = tf.placeholder(tf.float32, [64, None])
        parameters = forward.initialize_para()
        Y = forward.forward_prob(X, parameters, 1)
        Y_prediction = tf.argmax(Y)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(backward.Modelpath)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                Y_prediction = sess.run(Y_prediction, feed_dict={X: X_test})
                return Y_prediction
            else:
                print('No checkpoint file found')

def application():
    filepath = 'D:\\sa\\features\\TrainSet_20186501.h5'
    file = h5py.File(filepath, 'r')
    EEG_1 = file['EEG_1'][:]
    EEG_2 = file['EEG_2'][:]
    EEG = np.concatenate((EEG_1, EEG_2), axis=0)
    X = EEG[:, 92000:95000]
    file.close()
    Y_pre = restore_model(X)
    labelpath = 'D:\\sa\\features\\TrainSet_20186500.h5'
    file = h5py.File(labelpath, 'r')
    labels = file['train_set_num'][:]
    labels = np.squeeze(labels)
    Y = labels[92000:95000]
    #labels_one = backward.one_hot_matrix(labels, 6)
    Accuracy = np.mean((Y_pre[:] == Y[:]))
    tools.plot_hypnogram(Y_pre, Y, i = 2,mode = 0)
    print(Accuracy)
    print('test')


if __name__ == "__main__":
    application()

