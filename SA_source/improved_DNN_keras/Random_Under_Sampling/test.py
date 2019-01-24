import h5py
import numpy as np
import tensorflow as tf
import random
def one_hot_matrix(labels):
    C = tf.constant(5, name='C')
    one_hot_matrix = tf.one_hot(indices=labels, depth=C, axis=0)
    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()
    return one_hot


datapath = 'D:\\sa\\features\\TrainSet_20186501.h5'
labelpath = 'D:\\sa\\features\\TrainSet_20180903.h5'
file = h5py.File(datapath, 'r')
EEG_1 = file['EEG_1'][:]
EEG_2 = file['EEG_2'][:]
'''
EEG_1 = EEG_1[9:30,:]
EEG_2 = EEG_2[9:30,:]
'''
file.close()
EEG = np.concatenate((EEG_1, EEG_2), axis=0)
mean = np.reshape(np.mean(EEG, axis=1), [EEG.shape[0], 1])
std = np.reshape(np.std(EEG, axis=1), [EEG.shape[0], 1])
EEG -= mean
EEG /= std
EEG = EEG.T
file = h5py.File(labelpath, 'r')
labels = file['train_set_num'][:]
file.close()
labels = np.squeeze(labels)
labels_one = one_hot_matrix(labels)
labels_one = labels_one.T
X_train = EEG[0: 80000, :]


file = h5py.File(labelpath, 'r')
labels = file['train_set_num'][:]
labels = np.squeeze(labels)
labels_one = one_hot_matrix(labels)
labels_one = labels_one.T[0:80000,:]


Y_train = np.argmax(labels_one, axis=1)
seq_w = np.where(Y_train == 4)[0]
seq_rem = np.where(Y_train == 3)[0]
seq_s1 = np.where(Y_train == 2)[0]
seq_s2 = np.where(Y_train == 1)[0]
seq_sws = np.where(Y_train == 0)[0]


m_w = 5500
m_rem = 5565
m_s1 = 2142
m_s2 = 6400
m_sws = 4188
X_train_random = np.zeros((23795, 64), dtype=np.float32)
a = seq_w[np.random.randint(0, 55243, m_w)]
X_train_random[0: m_w, :] = X_train[a, :]  # wake stage
X_train_random[m_w: m_w + m_rem, :] = X_train[seq_rem, :]  # wake + rem
X_train_random[m_w + m_rem: m_w + m_rem + m_s1, :] = X_train[seq_s1, :]  # wake + rem + s1
b = seq_s2[np.random.randint(0, 12862, m_s2)]
X_train_random[m_w + m_rem + m_s1: m_w + m_rem + m_s1 + m_s2, :] = X_train[b,:]  # wake + rem + s1 + s2
X_train_random[m_w + m_rem + m_s1 + m_s2: m_w + m_rem + m_s1 + m_s2 + m_sws, :] = X_train[seq_sws,
                                                                                  :]  # wake + rem + s1 + s2 + sws
Y_train_random = np.zeros(23795, dtype=np.float32)
Y_train_random[0: m_w] = 4
Y_train_random[m_w: m_w + m_rem] = 3
Y_train_random[m_w + m_rem: m_w + m_rem + m_s1] = 2
Y_train_random[m_w + m_rem + m_s1: m_w + m_rem + m_s1 + m_s2] = 1
Y_train_random[m_w + m_rem + m_s1 + m_s2: m_w + m_rem + m_s1 + m_s2 + m_sws] = 0
Y_train_random = one_hot_matrix(Y_train_random)
print('test')