import tensorflow as tf
import numpy as np
import h5py
import BN_forward
import BN_backward
import test
import tools

def restore_model(X_test):
    '''

    :param X: size must be [64*n], here n is number of samples you want to prediction.
    :return: sleep stage
    '''
    with tf.Graph().as_default() as g:
        X = tf.placeholder(tf.float32, [None, 64])
        parameters = BN_forward.initialize_para()
        Y = BN_forward.forward_prob(X, parameters, 1, is_training = False)
        Y_prediction = tf.argmax(Y, axis = 1)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(BN_backward.Modelpath)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                Y_prediction = sess.run(Y_prediction, feed_dict={X: X_test})
                return Y_prediction
            else:
                print('No checkpoint file found')
def see_confusion_matrix(Y_prediction, Y_real):
    matrix = tf.contrib.metrics.confusion_matrix(Y_real, Y_prediction)
    with tf.Session() as sess:
        print(str(tf.Tensor.eval(matrix)))

def application():
    filepath = 'D:\\sa\\features\\TrainSet_20186501.h5'
    file = h5py.File(filepath, 'r')
    EEG_1 = file['EEG_1'][:]
    EEG_2 = file['EEG_2'][:]
    EEG = np.concatenate((EEG_1, EEG_2), axis=0)
    mean = np.reshape(np.mean(EEG, axis=1), [EEG.shape[0], 1])
    std = np.reshape(np.std(EEG, axis=1), [EEG.shape[0], 1])
    EEG -= mean
    EEG /= std
    X = EEG[:, 80000:].T
    file.close()
    Y_pre = restore_model(X)
    labelpath = 'D:\\sa\\features\\TrainSet_20186500.h5'
    file = h5py.File(labelpath, 'r')
    labels = file['train_set_num'][:]
    labels = np.squeeze(labels)
    Y = labels[80000:].T
    #labels_one = backward.one_hot_matrix(labels, 6)
    Accuracy = np.mean((Y_pre[:] == Y[:]))
    see_confusion_matrix(Y, Y_pre)
    tools.plot_hypnogram(Y_pre, Y, mode = 0)
    print(Accuracy)
    print('test')
    print(np.sum(Y == 2))

if __name__ == "__main__":
    application()

