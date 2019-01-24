import numpy as np
from keras import regularizers
from keras import layers
from keras.layers import  LSTM, Dense, Activation
from keras.models import Sequential, Model, load_model
import tools
import h5py
import tensorflow as tf
#from tqdm import tqdm
import time
import gc
from keras import backend as K


def see_confusion_matrix(Y_prediction, Y_real):
    matrix = tf.contrib.metrics.confusion_matrix(Y_real, Y_prediction)
    with tf.Session() as sess:
        print(str(tf.Tensor.eval(matrix)))

def one_hot_matrix(labels):
    C = tf.constant(6, name='C')
    one_hot_matrix = tf.one_hot(indices=labels, depth=C, axis=0)
    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()
    return one_hot


class RNN_model():
    def __init__(self, data_dir, label_dir, model_dir, status = True, **kwargs):
        '''

        :param data_dir: path of dataset
        :param label_dir: path of labelset
        :param model_dir: path of model
        :param train_test: True, when training data. False, when make prediction
        :param kwargs:
        '''
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.model_dir = model_dir
        self.epochs = kwargs.get('epochs', 2000)
        self.lstm_layers = kwargs.get('lstm_layer', 2)
        self.dropout = kwargs.get('dropout', 0.5)
        self.batch_size = kwargs.get('batch_size', 64)
        self.time_size = kwargs.get('time_size', 6)
        self.train_size = kwargs.get('train.size')
        self.optimizer = kwargs.get('optimizer', 'Adam')
        self.num_neurons = kwargs.get('num_neurons', 80)
        self.loss = kwargs.get('loss', 'categorical_crossentropy')
        self.num_train = kwargs.get('num_train', 80000)
        self.num_features = kwargs.get('num_features', 64)
        self.keep_train = kwargs.get('keep_train', True)
        self.learning_rate = kwargs.get('learning_rate', 0.0001)

        if status == True:
            X_train, Y_train, X_test, Y_test = self.get_train_data()
            self.model_train_rnn(X_train, Y_train, X_test, Y_test)
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
                Y = np.argmax(Y, axis = 1)
                tools.plot_hypnogram(self.prediction, Y)
                see_confusion_matrix(Y, self.prediction)
                K.clear_session()
                gc.collect()

    def get_train_data(self):
        datapath = self.data_dir
        labelpath = self.label_dir
        file = h5py.File(datapath, 'r')
        EEG_1 = file['EEG_1'][:]
        EEG_2 = file['EEG_2'][:]
        file.close()
        EEG = np.concatenate((EEG_1, EEG_2), axis=0)
        mean = np.reshape(np.mean(EEG, axis=1), [EEG.shape[0], 1])
        std = np.reshape(np.std(EEG, axis=1), [EEG.shape[0], 1])
        EEG -= mean
        EEG /= std
        EEG = EEG.T
        (m, n_x) = EEG.shape
        EEG_new = np.zeros((m - 6 + 1, 6, n_x))
        for i in range(m - 6 + 1):
            EEG_new[i, :, :] = EEG[i:i + 6, :]
        X_train = EEG_new[0 : self.num_train, :, :]
        X_test = EEG_new[self.num_train:, :, :]
        file = h5py.File(labelpath, 'r')
        labels = file['train_set_num'][:]
        file.close()
        labels = np.squeeze(labels)
        labels_one = one_hot_matrix(labels)
        labels_one = labels_one.T
        labels_one_new = labels_one[self.time_size - 1 :, :]
        Y_train = labels_one_new[0:self.num_train, :]
        Y_test = labels_one_new[self.num_train:, :]
        return X_train, Y_train, X_test, Y_test

    def get_prediction_data(self):
        datapath = self.data_dir
        labelpath = self.label_dir
        file = h5py.File(datapath, 'r')
        EEG_1 = file['EEG_1'][:]
        EEG_2 = file['EEG_2'][:]
        file.close()
        EEG = np.concatenate((EEG_1, EEG_2), axis=0)
        mean = np.reshape(np.mean(EEG, axis=1), [EEG.shape[0], 1])
        std = np.reshape(np.std(EEG, axis=1), [EEG.shape[0], 1])
        EEG -= mean
        EEG /= std
        EEG = EEG.T
        (m, n_x) = EEG.shape
        EEG_new = np.zeros((m - 6 + 1, 6, n_x))
        for i in range(m - 6 + 1):
            EEG_new[i, :, :] = EEG[i:i + 6, :]
        X = EEG_new[self.start : self.end, :, :]
        file = h5py.File(labelpath, 'r')
        labels = file['train_set_num'][:]
        file.close()
        labels = np.squeeze(labels)
        labels_one = one_hot_matrix(labels)
        labels_one = labels_one.T
        labels_one_new = labels_one[self.time_size - 1:, :]
        Y = labels_one_new[self.start : self.end, :]
        return X, Y



    def model_build_rnn(self):

        model = Sequential()
        for i in range(self.lstm_layers - 1):
            model.add(LSTM(batch_input_shape = (None, self.time_size, self.num_features),
                           output_dim = self.num_neurons, recurrent_dropout = self.dropout, return_sequences = True))
        model.add(LSTM(batch_input_shape=(None, self.time_size, self.num_features),
                       output_dim = self.num_neurons, recurrent_dropout = self.dropout))
        model.add(Dense(6))
        model.add(Activation('softmax'))
        return model

    def model_train_rnn(self, X_train, Y_train, X_test, Y_test):
        print('RNN_model is training')
        _, _, n_x = X_train.shape
        if self.keep_train == False:
            print('train from start')
            model = self.model_build_rnn()
        else:
            model = load_model(self.model_dir)
            print('there existed already a trained model, continue training')
        model.compile(optimizer = self.optimizer, loss = self.loss, metrics = ['accuracy'])
        model.fit(X_train, Y_train, batch_size = self.batch_size, epochs = self.epochs, shuffle=False)
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



















