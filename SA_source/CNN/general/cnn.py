import tensorflow as tf
import numpy as np
import h5py
import CNN_model

#datapath = 'D:\\sa\\features\\TrainSet_2.h5'
i = 19
datapath = 'D:\\sa\\features\\TrainSet_20181010.h5'
modelpath = 'D:\\sa\\model\\CNN_general\\%s.h5'%i
kwargs = {'epochs': 25, 'keep_train': False, 'batch_size': 128, 'seed_num': i}
model = CNN_model.CNN_model(datapath, modelpath, status = False, **kwargs)
