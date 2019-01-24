import tensorflow as tf
import numpy as np
import h5py
import CNN_model

#datapath = 'D:\\sa\\features\\TrainSet_2.h5'
datapath = 'D:\\sa\\features\\TrainSet_20180903.h5'
modelpath = 'D:\\sa\\model\\CNN_New_standard\\CNN_weighted_model_dropout0.2_30_v2.h5'
kwargs = {'epochs': 30, 'keep_train': False, 'batch_size': 128}
model = CNN_model.CNN_model(datapath, modelpath, status = True, **kwargs)
