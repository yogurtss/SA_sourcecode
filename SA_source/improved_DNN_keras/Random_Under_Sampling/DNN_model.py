import numpy as np
import h5py
import tools
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras.models import Sequential, Model, load_model
import gc
from keras import backend as K
import tensorflow as tf
'''
using a weighted matrix to offset the unbalanced distribution of the train-data.
the weighted matrix has 2 different ways.
1. according to the ratio of the data number
2. according to the accuracy of the results.
'''





def see_confusion_matrix(Y_prediction, Y_real):
    matrix = tf.contrib.metrics.confusion_matrix(Y_real, Y_prediction)
    with tf.Session() as sess:
        print(str(tf.Tensor.eval(matrix)))

def one_hot_matrix(labels):
    C = tf.constant(5, name='C')
    one_hot_matrix = tf.one_hot(indices=labels, depth=C, axis=0)
    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()
    return one_hot

class DNN_model():
    def __init__(self, data_dir, label_dir, model_dir, status = True, **kwargs):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.model_dir = model_dir
        self.num_layers = kwargs.get('num_layers', 3)
        self.epochs = kwargs.get('epochs', 250)
        self.dropout = kwargs.get('dropout', 0.3)
        self.batch_size = kwargs.get('batch_size', 128)
        self.optimizer = kwargs.get('optimizer', 'Adam')
        self.num_neurons = kwargs.get('num_neurons', 80)
        self.loss = kwargs.get('loss', 'categorical_crossentropy')
        self.num_train = kwargs.get('num_train', 80000)
        self.num_features = kwargs.get('num_features', 64)
        self.keep_train = kwargs.get('keep_train', True)
        self.learning_rate = kwargs.get('learning_rate', 0.0001)

        if status == True:
            X_train, Y_train, X_test, Y_test = self.get_train_data()
            self.model_train_DNN(X_train, Y_train, X_test, Y_test)
            K.clear_session()
            gc.collect()
        if status == False:
            if self.model_dir == None:
                raise FileNotFoundError('your model file is wrong, please check your path')
            else:
                self.start = int(input('please input your start sample(0-10097)'))
                self.end = int(input('please input your end sample'))
                X, Y = self.get_prediction_data()
                self.prediction, self.Accuracy = self.model_predict(X, Y)
                Y = np.argmax(Y, axis=1)
                see_confusion_matrix(Y, self.prediction)
                tools.plot_hypnogram(self.prediction, Y, mode=2)
                K.clear_session()
                gc.collect()

    def get_train_data(self):
        datapath = self.data_dir
        labelpath = self.label_dir
        file = h5py.File(datapath, 'r')
        EEG_1 = file['EEG_1'][:]
        EEG_2 = file['EEG_2'][:]
        '''
        EEG_1 = EEG_1[9:30,:]
        EEG_2 = EEG_2[9:30,:]
        '''
        file.close()
        EEG = np.concatenate((EEG_1, EEG_2), axis=0)
        mean = np.reshape(np.mean(EEG, axis=1), [EEG.shape[0], 1])
        std = np.reshape(np.std(EEG, axis=1), [EEG.shape[0], 1])
        EEG -= mean
        EEG /= std
        EEG = EEG.T
        file = h5py.File(labelpath, 'r')
        labels = file['train_set_num'][:]
        labels = np.squeeze(labels)
        labels_one = one_hot_matrix(labels)
        labels_one = labels_one.T
        file.close()
        X_train = EEG[0:self.num_train,:]
        X_test = EEG[self.num_train:,:]
        Y_train =  labels_one[0:self.num_train,:]
        Y_test = labels_one[self.num_train:,:]
        return X_train, Y_train, X_test, Y_test

    def get_prediction_data(self):
        datapath = self.data_dir
        labelpath = self.label_dir
        file = h5py.File(datapath, 'r')
        EEG_1 = file['EEG_1'][:]
        EEG_2 = file['EEG_2'][:]
        '''
        EEG_1 = EEG_1[9:30,:]
        EEG_2 = EEG_2[9:30,:]
        '''
        file.close()
        EEG = np.concatenate((EEG_1, EEG_2), axis=0)
        mean = np.reshape(np.mean(EEG, axis=1), [EEG.shape[0], 1])
        std = np.reshape(np.std(EEG, axis=1), [EEG.shape[0], 1])
        EEG -= mean
        EEG /= std
        EEG = EEG.T
        file = h5py.File(labelpath, 'r')
        labels = file['train_set_num'][:]
        file.close()
        labels = np.squeeze(labels)
        labels_one = one_hot_matrix(labels)
        labels_one = labels_one.T
        X = EEG[self.start: self.end, :]
        Y = labels_one[self.start: self.end, :]
        return X, Y

    def get_sequence(self, X_train ,Y_train):
        Y_train = np.argmax(Y_train, axis=1)
        seq_w = np.where(Y_train == 4)[0]
        seq_rem = np.where(Y_train == 3)[0]
        seq_s1 = np.where(Y_train == 2)[0]
        seq_s2 = np.where(Y_train == 1)[0]
        seq_sws = np.where(Y_train == 0)[0]
        return seq_w, seq_rem, seq_s1, seq_s2, seq_sws

    def random_under_sampling(self, X_train, Y_train, seq_w, seq_rem, seq_s1, seq_s2, seq_sws):
        m_w = 5500
        m_rem = 5565
        m_s1 = 2142
        m_s2 = 6400
        m_sws = 4188
        X_train_random = np.zeros((23795,64), dtype = np.float32)
        #X_train_random[0 : m_w, :] = X_train[seq_w[np.random.randint(0, 55243, m_w)], :]   #wake stage maybe something wrong TAT
        X_train_random[0: m_w, :] = X_train[seq_w[np.random.choice(np.arange(55243), size = m_w, replace = False)], :]
        X_train_random[m_w : m_w + m_rem, :] = X_train[seq_rem, :]                         #wake + rem
        X_train_random[m_w + m_rem : m_w + m_rem + m_s1, :] = X_train[seq_s1, :]           #wake + rem + s1
        #X_train_random[m_w + m_rem + m_s1 : m_w + m_rem + m_s1 + m_s2, :] = X_train[seq_s2[np.random.randint(0, 12862, m_s2)], :] #wake + rem + s1 + s2
        X_train_random[m_w + m_rem + m_s1: m_w + m_rem + m_s1 + m_s2, :] = X_train[
                                                                           seq_s2[np.random.choice(np.arange(12862), size = m_s2, replace = False)], :]
        X_train_random[m_w + m_rem + m_s1 + m_s2 : m_w + m_rem + m_s1 + m_s2 + m_sws, :] = X_train[seq_sws, :] #wake + rem + s1 + s2 + sws
        Y_train_random = np.zeros(23795, dtype = np.float32)
        Y_train_random[0 : m_w] = 4
        Y_train_random[m_w : m_w + m_rem] = 3
        Y_train_random[m_w + m_rem : m_w + m_rem + m_s1] = 2
        Y_train_random[m_w + m_rem + m_s1 : m_w + m_rem + m_s1 + m_s2] = 1
        Y_train_random[m_w + m_rem + m_s1 + m_s2 : m_w + m_rem + m_s1 + m_s2 + m_sws] = 0
        Y_train_random = one_hot_matrix(Y_train_random).T
        return X_train_random, Y_train_random



    def model_build_DNN(self):
        model = Sequential()
        model.add(Dense(self.num_neurons, input_dim= 64, activation='elu', kernel_initializer='he_normal'))
        #model.add(Dense(self.num_neurons, input_dim= 42, activation='elu', kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout))
        for i in range(self.num_layers - 1):
            model.add(Dense(self.num_neurons, input_dim = 80, activation= 'elu', kernel_initializer= 'he_normal'))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout))
        model.add(Dense(5, activation='softmax'))
        return model

    def model_train_DNN(self, X_train, Y_train, X_test, Y_test):
        print('DNN_model is training')
        _, n_x = X_train.shape

        if self.keep_train == False:
            print('train from start')
            model = self.model_build_DNN()
        else:
            model = load_model(self.model_dir)
            print('there existed already a trained model, continue training')


        model.compile(optimizer=self.optimizer, loss='weighted_categorical_crossentropy', metrics=['accuracy'])
        #model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
        seq_w, seq_rem, seq_s1, seq_s2, seq_sws = self.get_sequence(X_train, Y_train)
        for i in range(self.epochs):
            print('total epochs %s'%i)
            X_train_random, Y_train_random = self.random_under_sampling(X_train, Y_train, seq_w, seq_rem, seq_s1, seq_s2, seq_sws)
            model.fit(X_train_random, Y_train_random, batch_size=self.batch_size, epochs=1, shuffle=True)
        preds_0 = model.evaluate(X_train, Y_train)
        print('loss for train:' + str(preds_0[0]))
        print('evaluation for train:' + str(preds_0[1]))
        preds = model.evaluate(X_test, Y_test)
        print('loss for test:' + str(preds[0]))
        print('evaluation fot test:' + str(preds[1]))
        model.save(self.model_dir)

    def model_predict(self, X, Y):
        model = load_model(self.model_dir)
        Y_prediction = model.predict(X)
        Y_prediction = np.argmax(Y_prediction, axis = 1)
        predict = model.evaluate(X, Y)
        Accuracy = str(predict[1])
        print('Accuracy for prediction:' + Accuracy)
        return Y_prediction, Accuracy






