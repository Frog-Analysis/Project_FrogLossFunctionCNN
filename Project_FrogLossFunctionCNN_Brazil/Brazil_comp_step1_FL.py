# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 21:17:19 2021

@author: Administrator
"""

# performance comparison for clean recordings
import os
import pandas as pd
import numpy as np

cm_method, ack, class_acc_method = [], [], []
name_method = []

base_folder = './0421/1D_2D_Brazil_org_focal_1e-4'
loss_list = os.listdir(base_folder)
for loss_folder in loss_list:
    print(loss_folder)
    name_method.append(loss_folder)
    
    # ack_folder = os.path.join(base_folder, loss_folder, 'result_all_percent_0.8_winsize_8820_winover_0.8')
    ack_folder = os.path.join(base_folder, loss_folder, 'result_all_percent_0.8_winsize_0.2_winover_0.8')
    ack_path = ack_folder + '/accuracy_fscore_kappa.csv'
    
    ack_value = pd.read_csv(ack_path, low_memory=False, header=None)
    ack_value = ack_value.values
    # print(ack_value)

    ack.append(ack_value[1])

    cm_path = ack_folder + '/confusion_matrix.csv'
    cm_score = pd.read_csv(cm_path, header=None)   
    cm_method.append(cm_score.astype(int))
    
    cm_matrix = cm_score.values
    
    class_acc = cm_matrix.diagonal()/cm_matrix.sum(axis=1)
    class_acc_method.append(class_acc)


ack_mat = np.hstack(ack)
print('====max acc==')
print(np.max(ack_mat))
print(name_method[np.argmax(ack_mat)])

# class_acc_matrix = np.vstack(class_acc_method)
# class_acc_matrix = class_acc_matrix.T




