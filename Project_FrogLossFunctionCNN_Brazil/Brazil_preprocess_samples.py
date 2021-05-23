# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 22:40:50 2021

@author: Administrator
"""
import os
import librosa
import numpy as np
import pandas as pd

#--samples
cnt_train_sample, cnt_test_sample = [], []

base_folder = r'.\Brazil-Frog\0329_raw_data_clean\percent_0.8_winsize_8820_winover_0.8'
frog_list = os.listdir(base_folder)
for frog_name in frog_list:
    print(frog_name)
    
    frog_folder = os.path.join(base_folder, frog_name)
    
    train_path = os.path.join(frog_folder, 'train.csv')
    
    train_data = pd.read_csv(train_path, header=None)
    train_row, train_col = train_data.shape
    
    cnt_train_sample.append(train_row)
    
    test_path = os.path.join(frog_folder, 'test.csv')
    test_data = pd.read_csv(test_path, header=None)
    test_row, test_col = test_data.shape

    cnt_test_sample.append(test_row)

print('Total samples of Train data:' + str(np.sum(cnt_train_sample)))
print('Total samples of Test data:' + str(np.sum(cnt_test_sample)))

