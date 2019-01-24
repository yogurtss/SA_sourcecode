import h5py
import numpy as np
import tensorflow as tf
import random
#seed_num = 1
#np.random.seed(seed_num)
seq_train = np.random.choice(np.arange(40), size=32, replace = False)
num_train = np.zeros(80000)
num_entire = np.arange(0, 100942)
for i in range(32):
    num_train[i * 2500: (i + 1) * 2500] = np.arange(seq_train[i] * 2500, seq_train[i] * 2500 + 2500)
num_test = np.setdiff1d(num_entire, num_train)
num_train = num_train.astype(np.int)
num_test = num_test.astype(np.int)
a = np.max(num_train)
print('test')