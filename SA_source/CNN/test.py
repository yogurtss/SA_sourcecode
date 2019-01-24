import h5py
datapath = 'D:\\sa\\features\\TrainSet_20180903.h5'
file = h5py.File(datapath, 'r')
EEG_1 = file['EEG_1'][:].T
EEG_2 = file['EEG_2'][:].T
print('test')