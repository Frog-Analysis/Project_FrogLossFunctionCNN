# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 10:15:31 2018

fuse 1d, 2d, and 1d-2d

@author: Jie
"""
from __future__ import print_function

import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, cohen_kappa_score

# read out_put of 1D and 2D

method_prob_cell = []
method_overall_cell = []
method_class_cell = []
method_label_cell = []

baseFolder = r'F:\project_code\Project_FrogLossFunctionCNN_Brazil\0408'
baseFolder = baseFolder.replace('\\', '/')
method_list = os.listdir(baseFolder)
for method in method_list:
    
    method_folder = os.path.join(baseFolder, method, 'weak_my_cross_entropy_clean',
                                 'result_all_percent_0.8_winsize_8820_winover_0.8')
        
    method_prob = pd.read_csv(method_folder + '/predict_label_prob.csv', header=None, names=None)
    method_prob = method_prob.values  
    method_prob_cell.append(method_prob)
    
    method_overall = pd.read_csv(method_folder + '/accuracy_fscore_kappa.csv', header=None, names=None)
    method_overall = method_overall.values
    method_overall_cell.append(method_overall)
    
    method_class = pd.read_csv(method_folder + '/f1_score_class.csv', header=None, names=None)
    method_class = method_class.values
    method_class_cell.append(method_class)

    method_label = pd.read_csv(method_folder + '/predict_label.csv', header=None, names=None)
    method_label = method_label.values  
    method_label_cell.append(method_label)   
    
    if method == '2D_Brazil_org':        
        testLabel = pd.read_csv(method_folder + '/testLabel.csv', header=None, names=None)
        testLabel = testLabel.values     

n_samples = method_label.shape[0]

########################Feature#############################        
nClass = 11

out_label = []
for i_sample in range(n_samples):
    
    prob0 = method_prob_cell[0]
    prob1 = method_prob_cell[1]
    prob2 = method_prob_cell[2]
    # prob3 = method_prob_cell[3]

    ##############################################
    select_CNN = np.reshape(prob0[i_sample,:], [1,nClass])
    select_acoustic = np.reshape(prob1[i_sample,:], [1,nClass])
    select_Visual = np.reshape(prob2[i_sample,:], [1,nClass])
    # select_subnet = np.reshape(prob3[i_sample,:], [1,nClass])

    ##############class####################
    CNN_class = method_class_cell[0]
    acoustic_class = method_class_cell[1]
    Visual_class = method_class_cell[2]
    # subnet_class = method_class_cell[3]

    idx = 0
    select_CNN = np.multiply(select_CNN, CNN_class[idx])
    select_acoustic = np.multiply(select_acoustic, acoustic_class[idx])
    select_Visual = np.multiply(select_Visual, Visual_class[idx])
    # select_subnet = np.multiply(select_subnet, subnet_overall[idx])    
    
    
    ##############overall#####################
    # CNN_overall = method_overall_cell[0]
    # acoustic_overall = method_overall_cell[1]
    # Visual_overall = method_overall_cell[2]
    # # subnet_overall = method_overall_cell[3]

    # idx = 0
    # select_CNN = np.multiply(select_CNN, CNN_overall[idx])
    # select_acoustic = np.multiply(select_acoustic, acoustic_overall[idx])
    # select_Visual = np.multiply(select_Visual, Visual_overall[idx])
    # # select_subnet = np.multiply(select_subnet, subnet_overall[idx])
            
    ##############################################        
    CNN_value = np.amax(select_CNN)
    CNN_loc = np.argmax(select_CNN)
    
    acoustic_value = np.amax(select_acoustic)
    acoustic_loc = np.argmax(select_acoustic)  

    Visual_value = np.amax(select_Visual)
    Visual_loc = np.argmax(select_Visual)

    # subnet_value = np.amax(select_subnet)
    # subnet_loc = np.argmax(select_subnet)

    ##############################################
    # all_value = np.array([CNN_value, Visual_value])
    # all_loc = np.array([CNN_loc, Visual_loc])
    
    # all_value = np.array([CNN_value, acoustic_value])
    # all_loc = np.array([CNN_loc, acoustic_loc])
    
    # all_value = np.array([CNN_value, subnet_value])
    # all_loc = np.array([CNN_loc, subnet_loc])
    
    # all_value = np.array([CNN_value, acoustic_value, subnet_value])
    # all_loc = np.array([CNN_loc, acoustic_loc, subnet_loc])
    
    all_value = np.array([CNN_value, acoustic_value, Visual_value])
    all_loc = np.array([CNN_loc, acoustic_loc, Visual_loc])
    
    # all_value = np.array([CNN_value, subnet_value, Visual_value])
    # all_loc = np.array([CNN_loc, subnet_loc, Visual_loc])

    # all_value = np.array([CNN_value, acoustic_value, Visual_value, subnet_value])
    # all_loc = np.array([CNN_loc, acoustic_loc, Visual_loc, subnet_loc])
   
 
    #if CNN_value >= Librosa_value:            
    #    out_label.append(CNN_loc)        
    #else:
    #    out_label.append(Librosa_loc)

    final_loc = np.argmax(all_value)
    out_label.append(all_loc[final_loc])    
    

predict_label_test = np.asarray(out_label)

temp_cm = confusion_matrix(testLabel, predict_label_test)
class_acc_fusion2 = temp_cm.diagonal()/temp_cm.sum(axis=1)
print(class_acc_fusion2)

accuracy_result = accuracy_score(testLabel, predict_label_test)
fscore_result = f1_score(testLabel, predict_label_test, average='weighted')    
f1_score_class = f1_score(testLabel, predict_label_test, average=None)
kapppa_score = cohen_kappa_score(testLabel, predict_label_test)
 
# save_path = folderPath_CNN + '/Fusion_out_label.csv'
# np.savetxt(save_path, out_label, delimiter=",", fmt='%s')




