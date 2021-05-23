# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 21:17:19 2021

@author: Administrator
"""

# performance comparison for clean recordings
import os
import pandas as pd
import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
import re


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

# final_acc_list = []
# final_fscore_list = []
# final_kappa_list = []

y1, y2, y3, y4 = [], [], [], []

nRepeat = 5
for idx in range(nRepeat):        
    repeat_name = 'repeat_' + str(idx)
    # print(repeat_name)

    cm_method, ack, class_acc_method = [], [], []
    name_method = []
    
    base_folder = './0429/1D_2D_Aus_org_noise/' + repeat_name
    noise_list = os.listdir(base_folder)
    noise_list = sorted_alphanumeric(noise_list)
    for noise in noise_list:
        noise_folder = os.path.join(base_folder, noise)    
        print(noise)
        
        loss_list = os.listdir(noise_folder)
        loss_list = loss_list[1:2]
        # loss_list = loss_list[0:1]
        for loss_folder in loss_list:
            name_method.append(noise)
            
            ack_folder = os.path.join(noise_folder, loss_folder, 'result_all_percent_0.8_winsize_0.2_winover_0.8')
            # ack_folder = os.path.join(noise_folder, loss_folder, 'result_all_percent_0.8_winsize_8820_winover_0.8')

            ack_path = ack_folder + '/accuracy_fscore_kappa.csv'
            
            print(ack_path)
            
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
    print(ack_mat)
    
    class_acc_matrix = np.vstack(class_acc_method)
    class_acc_matrix = class_acc_matrix.T
    
    #--f1-score
    fscore_mat = ack_mat.reshape(4,4)   
    acc_mat = fscore_mat
    
    n_cnn, n_loss = acc_mat.shape
               
    y1.append(acc_mat[0,:])
    y2.append(acc_mat[1,:]) 
    y3.append(acc_mat[2,:]) 
    y4.append(acc_mat[3,:]) 


y1_mean, y1_std = np.mean(y1, axis=0), np.std(y1, axis=0)
y2_mean, y2_std = np.mean(y2, axis=0), np.std(y2, axis=0)
y3_mean, y3_std = np.mean(y3, axis=0), np.std(y3, axis=0)
y4_mean, y4_std = np.mean(y4, axis=0), np.std(y4, axis=0)

x = np.arange(0,len(y1_mean)) 



fig, ax = plt.subplots()
       
plt.plot(x, y1_mean, 'o-', label='Pink noise', linewidth=2, markersize=8)
plt.fill_between(x, y1_mean-y1_std, y1_mean+y1_std,
    alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')

plt.plot(x, y2_mean, 's-', label='Rain noise', linewidth=2, markersize=8)
plt.fill_between(x, y2_mean-y2_std, y2_mean+y2_std,
    alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF')

plt.plot(x, y3_mean, 'D-', label='White noise', linewidth=2, markersize=8)
plt.fill_between(x, y3_mean-y3_std, y3_mean+y3_std,
    alpha=0.5, edgecolor='#3F7F4C', facecolor='#7EFF99')

plt.plot(x, y4_mean, '.-', label='Wind noise', linewidth=2, markersize=8)
plt.fill_between(x, y4_mean-y4_std, y4_mean+y4_std,
    alpha=0.5, edgecolor='#3F6F2C', facecolor='#089F99')


plt.legend()

loss_label = ['-5', '-10', '-15', '-20']
plt.xticks(x, loss_label, rotation=0)
plt.grid('on')

font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 15,
}

plt.xlabel('SNR', font2)
plt.ylabel('F1 score', font2)
plt.ylim(0, 1)
# plt.title('Masked and NaN data')
# plt.show()        
# plt.savefig('./plot/' + 'best_1D_2D_CNN_noise.png')

# print(loss_folder)














