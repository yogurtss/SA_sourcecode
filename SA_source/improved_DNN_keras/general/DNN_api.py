import h5py
import numpy as np
import tensorflow as tf
import DNN_model
'''
seed_num = 14
np.random.seed(seed_num)
seq_train = np.random.choice(np.arange(40), size=32, replace = False)
print(seq_train)
print(len(seq_train))
'''
i= 14
datapath = 'D:\\sa\\features\\TrainSet_20186501.h5'
labelpath = 'D:\\sa\\features\\TrainSet_20180903.h5'
modelpath = 'D:\\sa\\model\\New_standard\\\general\\%s.h5'%i
kwargs = {'epochs': 50, 'keep_train': False, 'dropout': 0.3, 'batch_size': 128, 'seed_num': i}
model = DNN_model.DNN_model(datapath, labelpath, modelpath, status = False, **kwargs)
