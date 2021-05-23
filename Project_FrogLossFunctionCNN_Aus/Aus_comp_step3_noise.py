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


cm_method, ack, class_acc_method = [], [], []
name_method = []

base_folder = './0421/1D_2D_Aus_org_noise'
noise_list = os.listdir(base_folder)
noise_list = sorted_alphanumeric(noise_list)
for noise in noise_list:
    noise_folder = os.path.join(base_folder, noise)    
    print(noise)
    
    loss_list = os.listdir(noise_folder)
    for loss_folder in loss_list:
        name_method.append(noise)
        
        ack_folder = os.path.join(noise_folder, loss_folder, 'result_all_percent_0.8_winsize_0.2_winover_0.8')
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
print(ack_mat)

class_acc_matrix = np.vstack(class_acc_method)
class_acc_matrix = class_acc_matrix.T

#--f1-score
fscore_mat = ack_mat.reshape(4,4)   
acc_mat = fscore_mat

n_cnn, n_loss = acc_mat.shape
           
x = np.arange(0,n_loss) 
y1 = acc_mat[0,:]
y2 = acc_mat[1,:]
y3 = acc_mat[2,:]
y4 = acc_mat[3,:]

fig, ax = plt.subplots()
       
plt.plot(x, y1, 'o-', color='lightgrey', label='Pink noise', linewidth=2, markersize=8)
plt.plot(x, y2, 's-', label='Rain noise', linewidth=2, markersize=8)
plt.plot(x, y3, 'D-', label='White noise', linewidth=2, markersize=8)
plt.plot(x, y4, 'D-', label='Wind noise', linewidth=2, markersize=8)
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
plt.ylim(0.4, 1)
# plt.title('Masked and NaN data')
# plt.show()        
plt.savefig('./plot/' + 'best_1D_2D_CNN_noise.png')
















