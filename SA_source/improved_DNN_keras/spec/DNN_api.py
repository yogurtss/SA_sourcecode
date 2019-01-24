import h5py
import numpy as np
import tensorflow as tf
import DNN_model

datapath = 'D:\\sa\\features\\TrainSet_20186501.h5'
labelpath = 'D:\\sa\\features\\TrainSet_20186500.h5'
modelpath = 'D:\\sa\\model\\DNN_model_spec\\3.h5'
kwargs = {'epochs': 100, 'keep_train': False, 'dropout': 0.3, 'batch_size': 128}
model = DNN_model.DNN_model(datapath, labelpath, modelpath, status = False, **kwargs)