import h5py
import numpy as np
import tensorflow as tf
import RNN_model

datapath = 'D:\\sa\\features\\TrainSet_20186501.h5'
labelpath = 'D:\\sa\\features\\TrainSet_20186500.h5'
modelpath = 'D:\\sa\\model\\RNN_model_250epochs\\RNN250epochs.h5'
kwargs = {'epochs': 250, 'keep_train': False, 'dropout': 0.3, 'batch_size': 128}
model = RNN_model.RNN_model(datapath, labelpath, modelpath, status = False, **kwargs)
