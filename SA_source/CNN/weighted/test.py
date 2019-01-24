import h5py
import numpy as np
import tools
datapath = 'D:\\sa\\features\\TrainSet_20181001.h5'
file = h5py.File(datapath, 'r')
EEG_1 = file['EEG_1'][:]
EEG_2 = file['EEG_2'][:]
psd_1 = file['EEG_fre_1'][:]
psd_2 = file['EEG_fre_2'][:]
psd_1 = psd_1.astype(np.float32)
psd_2 = psd_2.astype(np.float32)
labels = file['train_set_num'][:]
tools.Save_Data_2(EEG_1, EEG_2, psd_1, psd_2, labels, i = 20181010)
print('test')