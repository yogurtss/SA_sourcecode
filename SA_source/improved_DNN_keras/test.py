import h5py
import numpy as np

labelpath = 'D:\\sa\\features\\TrainSet_20180903.h5'
file = h5py.File(labelpath, 'r')
labels = file['train_set_num'][:]
labels = labels[0:80000]
print(np.sum(labels == 1))