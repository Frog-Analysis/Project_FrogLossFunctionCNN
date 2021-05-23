# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 14:43:48 2020

@author: arnou
"""

import os
import errno
import numpy as np
import librosa
import pandas as pd
import more_itertools

from sklearn.preprocessing import StandardScaler 


def segment_audio(signal, fs, win_size, win_step):
    print('segment')    
    win_data = list(more_itertools.windowed(signal, n=win_size, step=win_step))
    return(win_data)


def obtain_win_label_single(seg_label):
    seg_win_label= np.zeros((len(seg_label), 1))
    
    for iSeg in range(len(seg_label)):
        win_label_value = np.asarray(seg_label[iSeg])
        win_label_value[win_label_value == None] = 0
        
        print(win_label_value)

        if np.sum(win_label_value) / len(win_label_value) >= 0.5:
            seg_win_label[iSeg] = 1   
            
    return(seg_win_label)


def obtain_win_label(seg_label):
    print('windowed label array to lable value')
    seg_win_label_exist = np.zeros((len(seg_label), 1))
    seg_win_label_strong = np.zeros((len(seg_label), 1))
    seg_win_label_mid = np.zeros((len(seg_label), 1))
    seg_win_label_weak = np.zeros((len(seg_label), 1))
    for iSeg in range(len(seg_label)):
        win_label_value = np.asarray(seg_label[iSeg])
        win_label_value[win_label_value == None] = 0
        
        #print(win_label_value)
             
        if np.sum(win_label_value) > 0:
            seg_win_label_exist[iSeg] = 1
        if np.sum(win_label_value) / len(win_label_value) == 1:
            seg_win_label_strong[iSeg] = 1
        if np.sum(win_label_value) / len(win_label_value) >= 0.75:
            seg_win_label_mid[iSeg] = 1
        if np.sum(win_label_value) / len(win_label_value) >= 0.5:
            seg_win_label_weak[iSeg] = 1
              
    # combine labels
    seg_win_label_all = np.concatenate((seg_win_label_exist, seg_win_label_strong, 
                                        seg_win_label_mid, seg_win_label_weak), axis=1)    
    return(seg_win_label_all)


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
            
            
def signal_1d_to_2d(feat_1d, fs):
            
    winsize, winstep = 0.02, 0.01 # second
    feat_2d = []
    n_sample, n_feat = feat_1d.shape       
    for i_sample in range(n_sample):            
        row_signal =  feat_1d[i_sample,:]           
        row_signal = np.asfortranarray(row_signal)            
        S = librosa.feature.melspectrogram(y=row_signal, sr=fs, n_mels=120, 
                           hop_length=int(winstep*fs), n_fft=int(winsize*fs),
                           fmax=fs/2) 
        
        feat_2d.append(np.array(S).ravel())   
    row, col = S.shape    
    
    return feat_2d, row, col
    

def reshape_data_3d(feat, time_sequence, row, col):
    
    n_samples, n_data = feat.shape     
    feat_final = []
    for i_samples in range(n_samples):        
        temp_vector = feat[i_samples,:]               
        temp_img = temp_vector.reshape(time_sequence, row, col)    
        feat_final.append(temp_img[:,:,:,np.newaxis])
    feat_final = np.array(feat_final)
    
    return feat_final   


def reshape_data(feat, row, col):
    
    n_samples, n_data = feat.shape     
    feat_final = []
    for i_samples in range(n_samples):        
        temp_vector = feat[i_samples,:]    
        temp_img = temp_vector.reshape(row, col)    
        feat_final.append(temp_img[:,:,np.newaxis])
    feat_final = np.array(feat_final)
    
    return feat_final                  
            

def reshape_data_block(feat):
    
    n_samples, n_data = feat.shape     
    feat_final = []
    for i_samples in range(n_samples):        
        temp_vector = feat[i_samples,:]   
        
        temp_vector_block = np.split(temp_vector, 7)
        mat_block = []
        n_block = len(temp_vector_block)
        for i_block in range(n_block):            
            mat_block.append(temp_vector_block[i_block].reshape(17, 59))
            
        temp_img = np.vstack(mat_block)
                
        feat_final.append(temp_img[:,:,np.newaxis])
    feat_final = np.array(feat_final)
    
    return feat_final   


def read_data(dataPath):    
    data = pd.read_csv(dataPath, header=None)
    #data = dd.read_csv(dataPath, header=None)
    data = data.values    
    feat = data[:,0:-1]
    label = data[:,-1]    
    
    return feat,label


def normalize_data(feat):
    
    res = StandardScaler().fit_transform(feat)
    return res


def fast_zscore(my_data):
    
    nCol = my_data.shape[1]
    data_col_list = []
    for iCol in range(nCol):
        print(iCol)
        data_col = my_data[:,iCol]
        data_col_zscore = (data_col - data_col.mean())/data_col.std()        
        data_col_list.append(data_col_zscore)
    
    return(data_col_list)


def fast_min_max(my_data):
    
    nCol = my_data.shape[1]
    data_col_list = []
    for iCol in range(nCol):
        print(iCol)
        data_col = my_data[:,iCol]
        data_col_zscore = (data_col - np.min(data_col)) / (np.max(data_col) - np.min(data_col))        
        data_col_list.append(data_col_zscore)
    
    return(data_col_list)




