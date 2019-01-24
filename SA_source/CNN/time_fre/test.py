import h5py
import tools
datapath = 'D:\\sa\\features\\TrainSet_20180903.h5'
file = h5py.File(datapath, 'r')
EEG_1 = file['EEG_1'][:].T
EEG_2 = file['EEG_2'][:].T
Labels = file['train_set_num'][:]
labels = Labels[800:2000]
tools.plot_hypnogram_1(labels)
print('test')