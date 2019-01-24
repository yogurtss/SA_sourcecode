import h5py
import numpy as np
import tensorflow as tf
def one_hot_matrix(labels):
    C = tf.constant(5, name='C')
    one_hot_matrix = tf.one_hot(indices=labels, depth=C, axis=0)
    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()
    return one_hot
datapath = 'D:\\sa\\features\\TrainSet_20181010.h5'
print('prepare train data')
file = h5py.File(datapath, 'r')
EEG_1 = file['EEG_1'][:].T
EEG_2 = file['EEG_2'][:].T
EEG_1 /= 100
EEG_2 /= 100
psd_1 = file['EEG_fre_1'][:].T
psd_2 = file['EEG_fre_2'][:].T

'''
mean_1 = np.mean(EEG_1)
std_1 = np.std(EEG_1)
mean_2 = np.mean(EEG_2)
std_2 = np.std(EEG_2)
EEG_1 -= mean_1
EEG_1 /= std_1
EEG_2 -= mean_2
EEG_2 /= std_2
'''

labels = file['train_set_num'][:]
labels = np.squeeze(labels)
labels_one = one_hot_matrix(labels)
labels_one = labels_one.T
file.close()
m, n_x = EEG_1.shape
data = np.zeros((m, n_x, 4, 1))
data[:, :, 0, 0] = EEG_1
data[:, :, 1, 0] = EEG_2
data[:, :, 2, 0] = psd_1
data[:, :, 3, 0] = psd_2
print('123')