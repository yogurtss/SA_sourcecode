# -*- coding: utf-8 -*-
"""
Created on Wed May  9 19:51:58 2018

@author: He Jia
"""

import os
import mne
import numpy as np
from copy import deepcopy
import tools

class Dataloader(object):
    
    def __init__(self, file, preload = True, channels=None,
                 epoch_len = 3000, start=None, stop=None):
        if not os.path.isfile(file): raise FileNotFoundError( 'File {} not found'.format(file))
        self.header = None
        self.epoch_len = epoch_len
        self.start = start
        self.stop = stop
        self.file = file
        self.avi_channels = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'Resp oro-nasal', 'EMG submental', 'Temp rectal', 'STI 014']
        self.channels = []
        self.length = 0
        
        if preload == True:
            self.load()
    
    def load_header(self, filename = None):
        if filename is None: file = self.file
        header = mne.io.read_raw_edf(file)
        header.sfreq = header.info['sfreq']
        self.header = header
        return header
    
    def check_channels(self):
        channels = self.header.ch_names
        assert(channels == self.avi_channels)
        labels = []
        numbers = []
        for ch in channels:
            labels.append(ch)
            numbers.append(channels.index(ch))
        self.channels_ids = (numbers, labels)
        return self.channels_ids
    
    def load_data(self):
        numbers, labels = self.channels_ids
        data, _ = deepcopy(self.header[numbers,:])
        self.EEG_Fpz_Cz = data[0,:].squeeze()
        self.length = len(self.EEG_Fpz_Cz)
        self.EEG_Pz_Oz = data[1,:].squeeze()
        self.EOG_horizontal = data[2,:].squeeze()
        self.Resp_oro_nasal = data[3,:].squeeze()
        self.EMG_submental = data[4,:].squeeze()
        self.Temp_rectal = data[5,:].squeeze()
        self.STI_014 = data[6,:].squeeze()
        
    def preprocess(self):
        self.EEG_Fpz_Cz = tools.butter_bandpass_filter(self.EEG_Fpz_Cz, 0.15, self.header.sfreq)
        self.EEG_Pz_Oz = tools.butter_bandpass_filter(self.EEG_Pz_Oz, 0.15, self.header.sfreq)
        self.EOG_horizontal = tools.butter_bandpass_filter(self.EOG_horizontal, 0.15,self.header.sfreq)
        self.EMG_submental = tools.butter_bandpass_filter(self.EMG_submental, 10, self.header.sfreq)

    def load(self, file = None):
        if file is None: file = self.file
        self.load_header()
        self.check_channels()
        self.load_data()
        self.preprocess()


        
        
