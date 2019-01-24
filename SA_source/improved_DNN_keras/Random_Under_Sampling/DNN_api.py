import h5py
import numpy as np
import tensorflow as tf
import DNN_model

datapath = 'D:\\sa\\features\\TrainSet_20186501.h5'
labelpath = 'D:\\sa\\features\\TrainSet_20180903.h5'
modelpath = 'D:\\sa\\model\\New_standard\\\Random_Under_Sampling\\BN_weighted_Random_Under_SamplingV4.h5'
kwargs = {'epochs': 250, 'keep_train': False, 'dropout': 0.3, 'batch_size': 128}
model = DNN_model.DNN_model(datapath, labelpath, modelpath, status = False, **kwargs)