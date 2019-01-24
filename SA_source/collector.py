# -*- coding: utf-8 -*-
"""
Created on Thu May 10 23:04:07 2018

@author: 31618
"""
import os
import dataloader
import numpy as np
import tools
class datacollector(object):
    
    def __init__(self,signalfile,activityfile, data_len = None, EEG = True, 
                 EOG = False, EMG = False, Save_signal = True, 
                 Save_activity = True, Convert = True, duration = 30):
        self.folder_name = signalfile
        self.acfolder_name = activityfile
        self.EEG = EEG
        self.EOG = EOG
        self.EMG = EMG
        self.Hyp = Save_activity
        self.data_len = []
        self.duration = duration

        if Save_signal == True :
            self.save_data()
    
    def save_data(self):
        rootdir = self.folder_name
        list = os.listdir(rootdir)
        self.data = {}
        N = self.duration*100
        EEG_temp_1 = np.empty((N,0))
        EEG_temp_2 = np.empty((N,0))
        EOG_temp = np.empty((N,0))
        EMG_temp = np.empty((N,0))
        Tem_temp = np.empty((N,0))
        print("data loading")
        for i in range(0,len(list)):
            path = os.path.join(rootdir,list[i])
            if os.path.isfile(path):
                signal = dataloader.Dataloader(path)
                length = len(signal.EEG_Fpz_Cz)
                self.data_len.append(length)
                if self.EEG == True :
                    EEG_raw_1 = signal.EEG_Fpz_Cz
                    EEG_raw_1 = (np.reshape(EEG_raw_1,(int(EEG_raw_1.shape[0]/N),N))).T
                    EEG_temp_1 = np.concatenate((EEG_temp_1,EEG_raw_1), axis = 1)
                    EEG_raw_2 = signal.EEG_Pz_Oz
                    EEG_raw_2 = (np.reshape(EEG_raw_2,(int(EEG_raw_2.shape[0]/N),N))).T
                    EEG_temp_2 = np.concatenate((EEG_temp_2,EEG_raw_2), axis = 1)
                    Tem_raw = signal.Temp_rectal
                    Tem_raw = (np.reshape(Tem_raw,(int(Tem_raw.shape[0]/N),N))).T
                    Tem_temp = np.concatenate((Tem_temp,Tem_raw), axis = 1)
  
                if self.EOG == True:
                    EOG_raw = signal.EOG_horizontal
                    EOG_raw = (np.reshape(EOG_raw,(int(EOG_raw.shape[0]/N),N))).T
                    EOG_temp = np.concatenate((EOG_temp,EOG_raw), axis = 1)
                    
                if self.EMG == True:
                    EMG_raw = signal.EMG_submental
                    EMG_raw = (np.reshape(EMG_raw,(int(EMG_raw.shape[0]/N),N))).T
                    EMG_temp = np.concatenate((EMG_temp,EMG_raw), axis = 1)
        '''
        if self.Convert == True:
            N = self.duration*100
            EEG_temp_1 = np.array(EEG_temp_1)
            EEG_temp_2 = np.array(EEG_temp_2)
            Tem_temp = np.array(Tem_temp)
            EEG_temp_1 = (np.reshape(EEG_temp_1,(int(EEG_temp_1.shape[0]/N),N))).T
            EEG_temp_2 = (np.reshape(EEG_temp_2,(int(EEG_temp_2.shape[0]/N),N))).T
            Tem_temp = (np.reshape(Tem_temp,(int(Tem_temp.shape[0]/N),N))).T
        '''
        
        self.data["EEG_1"] = EEG_temp_1
        self.data["EEG_2"] = EEG_temp_2
        self.data["Tem"] = Tem_temp
        print("EEG Channel1 :",self.EEG)
        print("EEG Channel2 :",self.EEG)
        print("EMG Channel : ",self.EMG)
        print("EOG Channel :",self.EOG)
        if self.Hyp == True:
            self.save_Activity()
    
    def save_Activity(self):
        rootdir = self.acfolder_name
        list = os.listdir(rootdir)
        labels = []
        print("labels loading")
        for i in range(0,len(list)):
            path = os.path.join(rootdir,list[i])
            if os.path.isfile(path):
                temp = tools.show_hypnogram(path)
                labels_temp = temp['description']
                labels.extend(labels_temp)
        print("labels :",self.Hyp)
        N = len(labels)
        #labels = np.reshape(np.array(labels),(N,1))
        self.data["Labels"] = labels
        return self.data
    

            
            
        
                
    
    
                            
        
                            
           
        
        
                
            