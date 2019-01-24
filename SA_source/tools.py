# -*- coding: utf-8 -*-
"""
Created on Mon May 14 14:10:20 2018

@author: 31618
"""

import numpy as np
import os
import matplotlib 
import matplotlib.pyplot as plt
import time
import  xml.dom.minidom
import h5py
from scipy.signal import butter, lfilter
from scipy import signal


def show_hypnogram(filename, window_len = 30):
    dom = xml.dom.minidom.parse(filename)
    data_hypnogram = {}
    description_temp = []
    description = []
    duration = []
    onset = []
    a = dom.getElementsByTagName('description')
    b = dom.getElementsByTagName('duration')
    c = dom.getElementsByTagName('onset')
    for n in a:
        description_temp.append(n.firstChild.data)
    description_temp.pop(-1)   
    
    for n in b:
        duration.append(int(n.firstChild.data))
    duration.pop(-1)   
    for n in c:
        onset.append(n.firstChild.data)
    onset.pop(-1)
    
    
    N = len(description_temp)
    
    for i in range(N):
        for j in range(int(duration[i]/window_len)):
            description.append(description_temp[i])
            
    
    

    data_hypnogram["description"] = description
    data_hypnogram["duration"] = duration
    data_hypnogram["onset"] = onset
    return data_hypnogram

def Convert_StageToNumber(labels, mode = True):
    #labels = labels.tolist()
    N = len(labels)
    if mode == True:
        labels = [word.replace('Sleep stage W','5').replace('Sleep stage R','4').replace('Movement time','5').replace('Sleep stage 1','3').replace('Sleep stage 2','2').replace('Sleep stage 3','1').replace('Sleep stage 4','0').replace('Sleep stage ?', '-1') for word in labels]
    else:
        labels = [word.replace('Sleep stage W','1').replace('Sleep stage R','0')
        .replace('Movement time','0').replace('Sleep stage 1','0').replace('Sleep stage 2','0')
        .replace('Sleep stage 3','0').replace('Sleep stage 4','0').replace('Sleep stage ?', '0') for word in labels]
    
    labels = np.reshape(np.array(list(map(int,labels))),(N,1))
    return labels
    
        

def Save_Data(data_1, data_2, data_3, labels, i):
    file = h5py.File('TrainSet_%s.h5' %str(i),'w')
    # 写入
    file.create_dataset('EEG_1', data = data_1)
    file.create_dataset('EEG_2', data = data_2)
    file.create_dataset('Tem', data = data_3)
    file.create_dataset('train_set_num',data = labels)
    file.close()

def Save_Data_1(data_1, data_2, labels, i):
    file = h5py.File('TrainSet_%s.h5' %str(i),'w')
    # 写入
    file.create_dataset('EEG_1', data = data_1)
    file.create_dataset('EEG_2', data = data_2)
    file.create_dataset('train_set_num',data = labels)
    file.close()
def Save_Data_2(data_1, data_2, data_3, data_4, labels, i):
    file = h5py.File('TrainSet_%s.h5' %str(i),'w')
    # 写入
    file.create_dataset('EEG_1', data = data_1)
    file.create_dataset('EEG_2', data = data_2)
    file.create_dataset('EEG_fre_1', data = data_3)
    file.create_dataset('EEG_fre_2', data = data_4)
    file.create_dataset('train_set_num', data=labels)
    file.close()

def Save_features(data_1, data_2, i):
    file = h5py.File('TrainSet_%s.h5' % str(i), 'w')
    file.create_dataset('EEG_1', data=data_1)
    file.create_dataset('EEG_2', data=data_2)
    file.close()

def Read_Data(filepath_1, filepath_2, labels_path):
    channel_1 = np.loadtxt(open(filepath_1,"rb"), delimiter = ",", skiprows = 0)
    channel_2 = np.loadtxt(open(filepath_2,"rb"), delimiter = ",", skiprows = 0)
    labels = np.loadtxt(open(labels_path,"rb"), delimiter = ",", skiprows = 0)
    return channel_1, channel_2, labels

def butter_bandpass(lowcut, highpass, fs, order=4):
    nyq = 0.5 * fs
    #       low = lowcut / nyq
    high = highpass / nyq
    b, a = butter(order, high, btype='highpass')
    return b, a

def butter_bandpass_filter(data, highpass, fs = 100, order=4):
    b, a = butter_bandpass(0, highpass, fs, order=order)
    y = lfilter(b, a, data)
    return y

def compute_derivation(data):
    '''
    compute the y*(dy/dt) for the Hjorth complexity
    :param data: signal y(t) -- array
    :return:
    '''
    #data_derivation = np.concatenate(np.zeros(((1,data.shape[1])), np.diff(data, axis = 0)), axis = 0)
    #data_derivation = np.concatenate((np.zeros((1, data.shape[1])), np.diff(data, axis = 0)), axis = 0)
    data_derivation = np.concatenate((np.reshape(data[0,:], (1, data.shape[1])), np.diff(data, axis=0)), axis=0)
    M = np.multiply(data, data_derivation)
    return M

def compute_Hjorth(data, Variance):
    M2 = compute_derivation(data)
    Var_M2 = np.var(M2, axis = 0)
    Mobility = np.sqrt(Var_M2 / Variance)
    M4 = compute_derivation(M2)
    Var_M4 = np.var(M4, axis = 0)
    Complexity = np.sqrt(Var_M4 / Var_M2) / Mobility
    return Mobility, Complexity

def compute_dominant_fre(psd,f):
    index = np.argmax(psd, axis = 0)
    dominant_fre = f[index]
    dominant_fre = np.reshape(dominant_fre, (1,psd.shape[1]))
    return dominant_fre

def compute_EDF(psd, f):
    m = psd.shape[1]
    limit = np.reshape(np.sum(psd, axis = 0), (1,m)) * 0.95
    index = np.argmax( (np.cumsum(psd, axis = 0) - limit) > 0, axis = 0)
    SEF = f[index]
    SEF = np.reshape(SEF, (1,m))
    return SEF

def compute_spectral_moments(psd, f):
    # ZCPS: zero crossings per second
    # EPS: extrema per second
    ZCPS = np.sqrt(np.sum(np.multiply(np.square(f), psd), axis = 0) / np.sum(psd, axis = 0)) / np.pi
    EPS = np.sqrt(np.sum(np.multiply(np.square(f), psd), axis = 0 )) / np.pi
    return ZCPS, EPS

def compute_bandpower(psd):
    # bandpower in 0.2-40Hz, notice, psd is already without DC-level,so corresponding frequency for psd[0] is 1/N*fs = 1/30
    band_power = np.sum(psd[0:5,:], axis = 0)
    return band_power

def compute_power_ratios(psd):
    '''
    a : .2-4Hz      b : 4-8Hz       c : 8-12
    d : 12-16       e : 16-30Hz     f : 30-40
    :param psd:
    :return:
    '''
    ab = np.sum(psd[5:120,:], axis = 0) / np.sum(psd[120:240,:], axis = 0)
    ac = np.sum(psd[5:120,:], axis = 0) / np.sum(psd[240:360,:], axis = 0)
    ad = np.sum(psd[5:120,:], axis = 0) / np.sum(psd[360:480,:], axis = 0)
    ae = np.sum(psd[5:120,:], axis = 0) / np.sum(psd[480:900,:], axis = 0)
    af = np.sum(psd[5:120,:], axis = 0) / np.sum(psd[900:1200,:], axis = 0)

    bc = np.sum(psd[120:240,:], axis = 0) / np.sum(psd[240:360,:], axis = 0)
    bd = np.sum(psd[120:240,:], axis = 0) / np.sum(psd[360:480,:], axis = 0)
    be = np.sum(psd[120:240,:], axis = 0) / np.sum(psd[480:900,:], axis = 0)
    bf = np.sum(psd[120:240,:], axis = 0) / np.sum(psd[900:1200,:], axis = 0)

    cd = np.sum(psd[240:360,:], axis = 0) / np.sum(psd[360:480,:], axis = 0)
    ce = np.sum(psd[240:360,:], axis = 0) / np.sum(psd[480:900,:], axis = 0)
    cf = np.sum(psd[240:360,:], axis = 0) / np.sum(psd[900:1200,:], axis = 0)

    de = np.sum(psd[360:480,:], axis = 0) / np.sum(psd[480:900,:], axis = 0)
    df = np.sum(psd[360:480,:], axis = 0) / np.sum(psd[900:1200,:], axis = 0)

    ef = np.sum(psd[480:900,:], axis = 0) / np.sum(psd[900:1200,:], axis = 0)
    return ab, ac, ad, ae, af, bc, bd, be, bf, cd, ce, cf, de, df, ef

def feature_extraction(h5path, time_features = True, fre_features = True, Ent_fearures = True, Tem_feature = True):
    file = h5py.File(h5path, 'r')
    EEG_1 = file['EEG_1'][:]
    EEG_2 = file['EEG_2'][:]
    m = EEG_1.shape[1]
    N = EEG_1.shape[0]
    EEG_1_time = np.empty((9,m), dtype = np.float32)
    EEG_2_time = np.empty((9,m), dtype = np.float32)
    EEG_1_fre = np.empty((21,m), dtype = np.float32)
    EEG_2_fre = np.empty((21,m), dtype = np.float32)
    EEG_1_ent = np.empty((2,m), dtype = np.float32)
    EEG_2_ent = np.empty((2, m), dtype=np.float32)
    fs = 100
    if time_features == True:
        '''
        Mean, Max, Median, Kurtosis, Skewness, Zeros crossings, Hjorth mobility, Hjorth complexity
        '''
        Mean_1 = np.mean(EEG_1, axis = 0)
        Mean_2 = np.mean(EEG_2, axis = 0)
        EEG_1_time[0,:] = Mean_1
        EEG_2_time[0,:] = Mean_2
        Max_1 = np.max(EEG_1, axis = 0)
        Max_2 = np.max(EEG_2, axis = 0)
        EEG_1_time[1,:] = Max_1
        EEG_2_time[1,:] = Max_2
        Median_1 = np.median(EEG_1, axis = 0)
        Median_2 = np.median(EEG_2, axis = 0)
        EEG_1_time[2,:] = Median_1
        EEG_2_time[2,:] = Median_2
        Variance_1 = np.var(EEG_1, axis = 0)
        Variance_2 = np.var(EEG_2, axis = 0)
        EEG_1_time[3,:] = Variance_1
        EEG_2_time[3,:] = Variance_2
        Kurtosis_1 = np.sum(((EEG_1 - Mean_1) ** 4), axis = 0) / N / (Variance_1 ** 2)
        Kurtosis_2 = np.sum(((EEG_2 - Mean_2) ** 4), axis = 0) / N / (Variance_2 ** 2)
        EEG_1_time[4,:] = Kurtosis_1
        EEG_2_time[4,:] = Kurtosis_2
        Skewness_1 = np.sum(((EEG_1 - Mean_1) ** 3), axis = 0) / N / (np.std((EEG_1), axis = 0) ** 3)
        Skewness_2 = np.sum(((EEG_2 - Mean_2) ** 3), axis = 0) / N / (np.std((EEG_2), axis = 0) ** 3)
        EEG_1_time[5,:] = Skewness_1
        EEG_2_time[5,:] = Skewness_2
        for i in range(m):
            EEG_1_i = EEG_1[:,i]
            EEG_2_i = EEG_2[:,i]
            zero_crossings_EEG_1 = len(np.where(np.diff(np.sign(EEG_1_i)))[0])
            zero_crossings_EEG_2 = len(np.where(np.diff(np.sign(EEG_2_i)))[0])
            EEG_1_time[6,i] = zero_crossings_EEG_1
            EEG_2_time[6,i] = zero_crossings_EEG_2
        Mobility_1, Complexity_1 = compute_Hjorth(EEG_1, Variance_1)
        Mobility_2, Complexity_2 = compute_Hjorth(EEG_2, Variance_2)
        EEG_1_time[7,:] = Mobility_1
        EEG_1_time[8,:] = Complexity_1
        EEG_2_time[7,:] = Mobility_2
        EEG_2_time[8,:] = Complexity_2

    if fre_features == True:
        fft_1_raw = np.fft.fft(EEG_1, axis = 0)
        fft_2_raw = np.fft.fft(EEG_2, axis = 0)
        psd_1 = np.abs(fft_1_raw[1:1501,:]) ** 2 / N  # from 1 to N/2+1w without DC-level
        psd_2 = np.abs(fft_2_raw[1:1501,:]) ** 2 / N  #sane
        f = np.reshape(np.arange(1,1501), (1500,1)) *fs /N
        average_power_1 = np.sum(np.multiply(psd_1, f), axis = 0) / np.sum(psd_1, axis = 0)
        average_power_2 = np.sum(np.multiply(psd_2, f), axis = 0) / np.sum(psd_2, axis = 0)
        EEG_1_fre[0,:] = average_power_1
        EEG_2_fre[0,:] = average_power_2
        dominant_fre_1 = compute_dominant_fre(psd_1, f)
        dominant_fre_2 = compute_dominant_fre(psd_2, f)
        EEG_1_fre[1,:] = dominant_fre_1
        EEG_2_fre[1,:] = dominant_fre_2
        SEF_1 = compute_EDF(psd_1, f)
        SEF_2 = compute_EDF(psd_2, f)
        EEG_1_fre[2,:] = SEF_1
        EEG_2_fre[2,:] = SEF_2
        ZCPS_1, EPS_1 = compute_spectral_moments(psd_1, f)
        ZCPS_2, EPS_2 = compute_spectral_moments(psd_2, f)
        EEG_1_fre[3,:] = ZCPS_1
        EEG_2_fre[3,:] = ZCPS_2
        EEG_1_fre[4,:] = EPS_1
        EEG_2_fre[4,:] = EPS_2
        band_power_1 = compute_bandpower(psd_1)
        band_power_2 = compute_bandpower(psd_2)
        EEG_1_fre[5,:] = band_power_1
        EEG_2_fre[5,:] = band_power_2
        ab_1, ac_1, ad_1, ae_1, af_1, bc_1, bd_1, be_1, bf_1, cd_1, ce_1, cf_1, de_1, df_1, ef_1 = compute_power_ratios(psd_1)
        ab_2, ac_2, ad_2, ae_2, af_2, bc_2, bd_2, be_2, bf_2, cd_2, ce_2, cf_2, de_2, df_2, ef_2 = compute_power_ratios(psd_1)
        EEG_1_fre[6,:] = ab_1
        EEG_2_fre[6,:] = ab_2
        EEG_1_fre[7,:] = ac_1
        EEG_2_fre[7,:] = ac_2
        EEG_1_fre[8,:] = ad_1
        EEG_2_fre[8,:] = ad_2
        EEG_1_fre[9,:] = ae_1
        EEG_2_fre[9,:] = ae_2
        EEG_1_fre[10,:] = af_1
        EEG_2_fre[10,:] = af_2
        EEG_1_fre[11,:] = bc_1
        EEG_2_fre[11,:] = bc_2
        EEG_1_fre[12,:] = bd_1
        EEG_2_fre[12,:] = bd_2
        EEG_1_fre[13,:] = be_1
        EEG_2_fre[13,:] = be_2
        EEG_1_fre[14,:] = bf_1
        EEG_2_fre[14,:] = bf_2
        EEG_1_fre[15,:] = cd_1
        EEG_2_fre[15,:] = cd_2
        EEG_1_fre[16,:] = ce_1
        EEG_2_fre[16,:] = ce_2
        EEG_1_fre[17,:] = cf_1
        EEG_2_fre[17,:] = cf_2
        EEG_1_fre[18,:] = de_1
        EEG_2_fre[18,:] = de_2
        EEG_1_fre[19,:] = df_1
        EEG_2_fre[19,:] = df_2
        EEG_1_fre[20,:] = ef_1
        EEG_2_fre[20,:] = ef_2

    if Ent_fearures == True:
        psd_1_norm = psd_1 / np.sum(psd_1, axis = 0)
        PSE_1 = - np.sum(np.multiply(psd_1_norm, np.log(psd_1_norm)), axis = 0)
        psd_2_norm = psd_2 / np.sum(psd_2, axis = 0)
        PSE_2 = - np.sum(np.multiply(psd_2_norm, np.log(psd_2_norm)), axis = 0)
        EEG_1_ent[0,:] = PSE_1
        EEG_2_ent[0,:] = PSE_2
        PSD_1_mean = np.mean(psd_1_norm, axis = 0)
        PSD_2_mean = np.mean(psd_2_norm, axis = 0)
        PSE_1_relative = -np.sum(np.multiply(psd_1_norm, np.log(psd_1_norm / PSD_1_mean)), axis = 0)
        PSE_2_relative = -np.sum(np.multiply(psd_2_norm, np.log(psd_2_norm / PSD_2_mean)), axis = 0)
        EEG_1_ent[1,:] = PSE_1_relative
        EEG_2_ent[1,:] = PSE_2_relative
    return EEG_1_time, EEG_2_time, EEG_1_fre, EEG_2_fre, EEG_1_ent, EEG_2_ent

def plot_eeg(datapath, labelpath):
    num = int(input('please input the number you want to draw'))
    file = h5py.File(datapath, 'r')
    EEG_1 = file['EEG_1'][:].T
    EEG_2 = file['EEG_2'][:].T
    file.close()
    data_1 = EEG_1[num, :]
    data_2 = EEG_2[num, :]
    file = h5py.File(labelpath, 'r')
    labels = file['train_set_num'][:]
    file.close()
    label = str(labels[num, :])
    print('the sleep stage you want to draw is %s' %label)
    m = 3000
    x = np.arange(0, m, 1)
    plt.figure(figsize=[10,4])
    plt.subplot(211)
    plt.plot(x, data_1)
    plt.title('EEG(fpz-oz) signal for 30 duration')
    plt.subplot(212)
    plt.plot(x, data_2)
    plt.title('EEG(pz-oz) signal for 30 duration')
    now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    plt.savefig('D:\\sa\\EEG_FIGURE\\%s.jpg' % now)
    plt.show()

def plot_hypnogram(label_predict, label, mode = 0):
    m = len(label_predict)

    if m != len(label): raise ValueError
    if mode == 0:
        label_predict[label_predict == 0] = 1
        label[label == 0] = 1
        print('plot the hypnogram according to the new standard')
        print('assuming Wake, movement: 5 REM: 4 Stage 1: 3 Stage 2: 2 Stage 3: 1 Stage 4:0')
        x = np.arange(0,m,1)
        plt.figure(figsize=[10,6])
        plt.subplot(211)
        plt.plot(x , label_predict)
        plt.ylim(0,6)
        plt.title('Hypnogram for prediction')
        plt.yticks([5, 4, 3, 2, 1],
                   ['wake', 'REM', 'S1', 'S2', 'SWS'])
        plt.subplot(212)
        plt.plot(x , label)
        plt.ylim(0, 6)
        plt.title('Hypnogram for 7 hours')
        plt.yticks([5, 4, 3, 2, 1],
                   ['wake', 'REM', 'S1', 'S2', 'SWS'])
        now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
        plt.savefig('D:\\sa\\HYPfigures\\%s.jpg' %now)
        plt.show()
    if mode == 1:
        label_predict += 1
        label += 1
        print('plot the hypnogram according to the old standard')
        print('assuming Wake, movement: 5 REM: 4 Stage 1: 3 Stage 2: 2 Stage 3: 1 Stage 4:0')
        x = np.arange(0, m, 1)
        plt.figure(figsize=[10, 6])
        plt.subplot(211)
        plt.plot(x, label_predict)
        plt.ylim(0, 6)
        plt.title('Hypnogram for prediction')
        plt.yticks([5, 4, 3, 2, 1],
                   ['wake', 'REM', 'S1', 'S2', 'SWS'])
        plt.subplot(212)
        plt.plot(x, label)
        plt.ylim(0, 6)
        plt.title('Hypnogram for 7 hours')
        plt.yticks([5, 4, 3, 2, 1],
                   ['wake', 'REM', 'S1', 'S2', 'SWS'])
        now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
        plt.savefig('D:\\sa\\HYPfigures\\%s.jpg' % now)
        plt.show()
    if mode == 2:
        label_predict += 1
        label += 1
        print('plot the hypnogram according to the new standard without transform')
        print('assuming Wake, movement: 5 REM: 4 Stage 1: 3 Stage 2: 2 SWS: 1')
        x = np.arange(0, m, 1)
        plt.figure(figsize=[10, 6])
        plt.subplot(211)
        plt.plot(x, label_predict)
        plt.ylim(0, 6)
        plt.title('predicted Hypnogram for 5 hours')
        plt.yticks([5, 4, 3, 2, 1],
                   ['wake', 'REM', 'S1', 'S2', 'SWS'])
        plt.subplot(212)
        plt.plot(x, label)
        plt.ylim(0, 6)
        plt.xlabel('epochs (120 epochs for 1 hour)')
        plt.title('actual Hypnogram for 5 hours')
        plt.yticks([5, 4, 3, 2, 1],
                   ['wake', 'REM', 'S1', 'S2', 'SWS'])
        now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
        plt.savefig('D:\\sa\\HYPfigures\\%s.jpg' % now)
        plt.show()


def data_augmentation(h5path, stage_1 = True, stage_2 = False, stage_3 = True, stage_4 = True):
    file = h5py.File(h5path, 'r')
    EEG_1 = file['EEG_1'][:]
    EEG_2 = file['EEG_2'][:]
    Labels = file['train_set_num'][:]
    m = 80000
    EEG_1 = EEG_1[:, 0:80000]
    EEG_2 = EEG_2[:, 0:80000]
    Labels = Labels[0:80000, :]
    EEG_1_augment = np.empty([3000, 0])
    EEG_2_augment = np.empty([3000,0])
    if stage_1 == True:
        for i in range(m):
            print(i)
            if Labels[i] == 3 and Labels[i] == Labels[i + 1]:
                EEG_1_augment = np.column_stack((EEG_1_augment, np.concatenate((EEG_1[500:, i], EEG_1[0:500, i + 1]), axis=0)))
                EEG_1_augment = np.column_stack((EEG_1_augment, np.concatenate((EEG_1[1000:, i], EEG_1[0:1000, i + 1]), axis=0)))
                EEG_1_augment = np.column_stack((EEG_1_augment, np.concatenate((EEG_1[1500:, i], EEG_1[0:1500, i + 1]), axis=0)))
                EEG_1_augment = np.column_stack((EEG_1_augment, np.concatenate((EEG_1[2000:, i], EEG_1[0:2000, i + 1]), axis=0)))
                EEG_1_augment = np.column_stack((EEG_1_augment, np.concatenate((EEG_1[2500:, i], EEG_1[0:2500, i + 1]), axis=0)))

                EEG_2_augment = np.column_stack((EEG_2_augment, np.concatenate((EEG_2[500:, i], EEG_2[0:500, i + 1]), axis=0)))
                EEG_2_augment = np.column_stack((EEG_2_augment, np.concatenate((EEG_2[1000:, i], EEG_2[0:1000, i + 1]), axis=0)))
                EEG_2_augment = np.column_stack((EEG_2_augment, np.concatenate((EEG_2[1500:, i], EEG_2[0:1500, i + 1]), axis=0)))
                EEG_2_augment = np.column_stack((EEG_2_augment, np.concatenate((EEG_2[2000:, i], EEG_2[0:2000, i + 1]), axis=0)))
                EEG_2_augment = np.column_stack((EEG_2_augment, np.concatenate((EEG_2[2500:, i], EEG_2[0:2500, i + 1]), axis=0)))
        n_1 = EEG_1_augment.shape[1]
        Labels_augment_1 = 3 * np.ones(n_1)
        Labels_augment_1 = np.reshape(Labels_augment_1, [n_1,1])
        Labels_augment_1 = Labels_augment_1.astype(np.float32)

    if stage_3 == True:
        for i in range(m):
            print(i)
            if Labels[i] == 1 and Labels[i] == Labels[i + 1]:
                EEG_1_augment = np.column_stack((EEG_1_augment, np.concatenate((EEG_1[1000:, i], EEG_1[0:1000, i + 1]), axis=0)))
                EEG_1_augment = np.column_stack((EEG_1_augment, np.concatenate((EEG_1[2000:, i], EEG_1[0:2000, i + 1]), axis=0)))
                EEG_2_augment = np.column_stack((EEG_2_augment, np.concatenate((EEG_2[1000:, i], EEG_2[0:1000, i + 1]), axis=0)))
                EEG_2_augment = np.column_stack((EEG_2_augment, np.concatenate((EEG_2[2000:, i], EEG_2[0:2000, i + 1]), axis=0)))
        n_2 = EEG_1_augment.shape[1] - n_1
        Labels_augment_2 =  np.ones(n_2)
        Labels_augment_2 = np.reshape(Labels_augment_2, [n_2, 1])
        Labels_augment_2 = Labels_augment_2.astype(np.float32)
        Labels_augment = np.concatenate((Labels_augment_1, Labels_augment_2), axis = 0)

    if stage_4 == True:
        for i in range(m):
            print(i)
            if Labels[i] == 0 and Labels[i] == Labels[i + 1]:
                EEG_1_augment = np.column_stack((EEG_1_augment, np.concatenate((EEG_1[1000:, i], EEG_1[0:1000, i + 1]), axis=0)))
                EEG_1_augment = np.column_stack((EEG_1_augment, np.concatenate((EEG_1[2000:, i], EEG_1[0:2000, i + 1]), axis=0)))
                EEG_2_augment = np.column_stack((EEG_2_augment, np.concatenate((EEG_2[1000:, i], EEG_2[0:1000, i + 1]), axis=0)))
                EEG_2_augment = np.column_stack((EEG_2_augment, np.concatenate((EEG_2[2000:, i], EEG_2[0:2000, i + 1]), axis=0)))
        n_3 = EEG_1_augment.shape[1] - n_2 - n_1
        Labels_augment_3 =  np.zeros(n_3)
        Labels_augment_3 = np.reshape(Labels_augment_3, [n_3, 1])
        Labels_augment_3 = Labels_augment_3.astype(np.float32)
        Labels_augment = np.concatenate((Labels_augment, Labels_augment_3), axis = 0)
    return EEG_1_augment, EEG_2_augment, Labels_augment


def plot_hypnogram_1(label):
    m = len(label)
    label += 1
    print('plot the hypnogram according to the new standard')
    print('assuming Wake, movement: 5 REM: 4 Stage 1: 3 Stage 2: 2 Stage 3: 1 Stage 4:0')
    x = np.arange(0,m,1)
    plt.figure(figsize=[16,9])
    plt.plot(x , label)
    plt.ylim(0, 6)
    plt.xlim(0, m)
    plt.xlabel('epochs (120 epochs for 1 hour)')
    plt.ylabel('stages')
    plt.yticks([5, 4, 3, 2, 1],
               ['Wake', 'REM', 'S1', 'S2', 'SWS'])
    now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    plt.savefig('D:\\sa\\HYPfigures\\\\example\\%s.jpg' %now)
    plt.show()












    


    
    
    

    
    
        