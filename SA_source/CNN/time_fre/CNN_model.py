import numpy as np
import h5py
from keras.layers import Dense, BatchNormalization, Conv1D, MaxPool1D, Dropout, Flatten
from keras.models import Sequential, Model, load_model
from keras import regularizers
import tools
import h5py
import tensorflow as tf
import gc
from keras import backend as K
import keras

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

def shuffle_data(X, Y):
    index = np.arange(X.shape[0])
    np.random.shuffle(index)
    X = X[index, :, :]
    Y = Y[index,:]
    return X,Y

class CNN_model():
    def __init__(self, data_dir, model_dir, status = True, **kwargs):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.epochs = kwargs.get('epochs', 100)
        self.dropout_cnn = kwargs.get('dropout_cnn', 0.2)
        self.dropout_dnn = kwargs.get('dropout_dnn', 0.5)
        self.batch_size = kwargs.get('batch_size', 256)
        #self.optimizer = kwargs.get('optimizer', 'Adam')
        self.loss = kwargs.get('loss', 'categorical_crossentropy')
        self.num_train = kwargs.get('num_train', 80000)
        self.keep_train = kwargs.get('keep_train', True)
        self.learning_rate = kwargs.get('learning_rate', 0.0001)


        if status == True:
            X_train, Y_train, X_test, Y_test = self.get_train_data()
            self.model_train_cnn(X_train, Y_train, X_test, Y_test)
            K.clear_session()
            gc.collect()

        if status == False:
            if self.model_dir == None:
                raise FileNotFoundError('your model file is wrong, please check your path')
            else:
                self.start = int(input('please input your start sample(0-10097)'))
                self.end = int(input('please input your end sample'))
                X, Y = self.get_predict_data()
                print('2nd step')
                self.prediction, self.accuracy = self.model_predict(X, Y)
                Y = np.argmax(Y, axis = 1)
                tools.plot_hypnogram(self.prediction, Y, mode=1)
                see_confusion_matrix(Y, self.prediction)
                K.clear_session()
                gc.collect()


    def get_train_data(self):
        datapath = self.data_dir
        print('prepare train data')
        file = h5py.File(datapath, 'r')
        EEG_1 = file['EEG_1'][:].T
        EEG_2 = file['EEG_2'][:].T
        EEG_1 /= 100
        EEG_2 /= 100
        psd_1 = file['EEG_fre_1'][:].T
        psd_2 = file['EEG_fre_2'][:].T
        '''
        mean_1 = np.mean(EEG_1)
        std_1 = np.std(EEG_1)
        mean_2 = np.mean(EEG_2)
        std_2 = np.std(EEG_2)
        EEG_1 -= mean_1
        EEG_1 /= std_1
        EEG_2 -= mean_2
        EEG_2 /= std_2
        '''

        labels = file['train_set_num'][:]
        labels = np.squeeze(labels)
        labels_one = one_hot_matrix(labels)
        labels_one = labels_one.T
        file.close()
        m, n_x = EEG_1.shape
        EEG = np.zeros((m, n_x, 2))
        #EEG[:,:,0] = EEG_1
        #EEG[:,:,1] = EEG_2
        EEG[:,:,0] = psd_1
        EEG[:,:,1] = psd_2
        #mean = np.reshape(np.mean(EEG, axis=1), [EEG.shape[0], 1, 2])
        #std = np.reshape(np.std(EEG, axis=1), [EEG.shape[0], 1, 2])
        #EEG -= mean
        #EEG /= std
        X_train = EEG[0 : self.num_train, :, :]
        X_test = EEG[self.num_train : , :, :]
        Y_train = labels_one[0:self.num_train, :]
        Y_test = labels_one[self.num_train:, :]
        return X_train, X_test, Y_train, Y_test

    def get_predict_data(self):
        datapath = self.data_dir
        print('prepare predict data')
        file = h5py.File(datapath, 'r')
        EEG_1 = file['EEG_1'][:].T
        EEG_2 = file['EEG_2'][:].T
        print('1111')
        #mean_1 = np.mean(EEG_1)
        #std_1 = np.std(EEG_1)
        #mean_2 = np.mean(EEG_2)
        #std_2 = np.std(EEG_2)
        #EEG_1 -= mean_1
        #EEG_1 /= std_1
        #EEG_2 -= mean_2
        #EEG_2 /= std_2
        EEG_1 /= 100
        EEG_2 /= 100
        psd_1 = file['EEG_fre_1'][:].T
        psd_2 = file['EEG_fre_2'][:].T
        labels = file['train_set_num'][:]
        print('2222')
        labels = np.squeeze(labels)
        labels_one = one_hot_matrix(labels)
        labels_one = labels_one.T
        file.close()
        m, n_x = EEG_1.shape
        EEG = np.zeros((m, n_x, 2))
        #EEG[:,:,0] = EEG_1
        #EEG[:,:,1] = EEG_2
        EEG[:,:,0] = psd_1
        EEG[:,:,1] = psd_2
        X = EEG[self.start : self.end, :, :]
        Y = labels_one[self.start:self.end, :]
        print('finished')
        return X, Y


    def model_build_cnn(self, input_shape):

        model = Sequential()
        model.add(Conv1D(filters = 128, kernel_size = (50), strides = 5, padding = 'valid',
                         activation = 'elu', input_shape = input_shape, kernel_initializer = 'he_normal', ))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_cnn))
        model.add(Conv1D(filters = 256, kernel_size = (5), strides = 1, padding = 'valid',
                         activation= 'elu', kernel_initializer = 'he_normal'))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_cnn))
        model.add(MaxPool1D())
        model.add(Conv1D(filters = 300, kernel_size = (5), strides = 2, padding = 'valid',
                         activation = 'elu', kernel_initializer = 'he_normal'))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_cnn))
        model.add(MaxPool1D())
        model.add(Flatten())
        model.add(Dense(1500, activation = 'elu', kernel_initializer = 'he_normal'))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_dnn))
        model.add(Dense(1500, activation = 'elu', kernel_initializer = 'he_normal'))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_dnn))
        model.add(Dense(5,activation = 'softmax'))
        return model



    def model_train_cnn(self, X_train, X_test, Y_train, Y_test):
        _, n_x, ch = X_train.shape
        print(X_train.shape)
        input_shape = (n_x, ch)
        print('CNN_model is training')
        if self.keep_train == False:
            print('train from start')
            model = self.model_build_cnn(input_shape)
        else:
            model = load_model(self.model_dir)
            print('there existed already a trained model, continue training')
        self.optimizer = keras.optimizers.Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(optimizer = self.optimizer, loss = self.loss, metrics = ['accuracy'])
        model.fit(X_train, Y_train, batch_size = self.batch_size, epochs = self.epochs, shuffle=True)
        preds_0 = model.evaluate(X_train, Y_train)
        print('loss for train:' + str(preds_0[0]))
        print('evaluation for train:' + str(preds_0[1]))
        preds = model.evaluate(X_test, Y_test)
        print('loss for test:' + str(preds[0]))
        print('evaluation fot test:' + str(preds[1]))
        model.save(self.model_dir)

    def model_predict(self, X, Y):
        print('loading model')
        model = load_model(self.model_dir)
        print('start')
        Y_prediction = model.predict(X)
        Y_prediction = np.argmax(Y_prediction, axis = 1)
        predict = model.evaluate(X, Y)
        Accuracy = str(predict[1])
        print('Accuracy for prediction:' + Accuracy)
        return Y_prediction, Accuracy




