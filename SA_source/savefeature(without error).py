# -*- coding: utf-8 -*-
"""
Created on Thu May 31 00:20:18 2018

@author: yogurts
"""

import h5py
import numpy as np
import tools
path1 = 'D:\\sa\\features\\TrainSet_999.h5'
path2 = 'D:\\sa_matlab\\TrainSet_100.h5'
file = h5py.File(path2, 'r')
EEG_1 = file['EEG_1'][:]
EEG_2 = file['EEG_2'][:]
Tem = file['Tem'][:]
labels_0 = file['train_set_num'][:]
file.close()
error = np.arange(51681, 52483, 1).tolist()
EEG_1 = np.delete(EEG_1, error, axis = 1)
EEG_2 = np.delete(EEG_2, error, axis = 1)
Tem = np.delete(Tem, error, axis = 1)
labels = np.delete(labels_0, error, axis = 0)
index = np.argwhere(labels == -1)
tools.Save_Data(EEG_1, EEG_2, Tem, labels, i=20186500)
file = h5py.File(path1, 'r')
EEG_1 = file['EEG_1'][:]
EEG_2 = file['EEG_2'][:]
print('before', EEG_1.shape)
EEG_1 = np.delete(EEG_1, error, axis = 1)
EEG_2 = np.delete(EEG_2, error, axis = 1)
EEG = np.concatenate((EEG_1, EEG_2), axis = 0)
print('after ', EEG.shape)
print('EEG_1 after', EEG_1.shape)
EEG = np.delete(EEG, error, axis = 1)
tools.Save_features(EEG_1, EEG_2, i=20186501)


