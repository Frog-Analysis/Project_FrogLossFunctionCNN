# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 19:02:25 2021

@author: Administrator
"""

import os
import pandas as pd
import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from matplotlib import pyplot as pl

from MyClass_python import base_function as bf

#--figure 1
base_folder = r'.\0429\figure1'

"""
Loss function + CNN architecture

"""

final_acc_list = []
final_fscore_list = []
final_kappa_list = []

nRepeat = 5
for idx in range(nRepeat):        
    repeat_name = 'repeat_' + str(idx)
    # print(repeat_name)

    acc_list = []
    fscore_list = []
    kappa_list = []
    method_list = []
        
    cnn_arch_list = os.listdir(base_folder)
    for cnn_arch in cnn_arch_list:
        print(cnn_arch)        
        cnn_arch_folder = os.path.join(base_folder, cnn_arch, repeat_name)
        
        cnn_loss_list = os.listdir(cnn_arch_folder) 
        # cnn_loss_list = cnn_loss_list[0:3] + cnn_loss_list[4:]
        # print(cnn_loss_list)
        for cnn_loss in cnn_loss_list:            
            cnn_loss_path = os.path.join(cnn_arch_folder, cnn_loss)
            
            tmp_list = os.listdir(cnn_loss_path)
            for tmp_name in tmp_list:
                final_path = os.path.join(cnn_loss_path, tmp_name)
                # print(final_path)
                
                acf_data = pd.read_csv(os.path.join(final_path, 'accuracy_fscore_kappa.csv'), header=None)
                acf_value = acf_data.values
                
                acc_list.append(acf_value[0])
                fscore_list.append(acf_value[1])
                kappa_list.append(acf_value[2])
                
                method_list.append([cnn_arch, cnn_loss])
              
    acc_array = np.vstack(acc_list)
    fscore_array = np.vstack(fscore_list)
    kappa_array = np.vstack(kappa_list)

    final_acc_list.append(acc_array)
    final_fscore_list.append(fscore_array)
    final_kappa_list.append(kappa_array)


#############################################################    
#--calculate Mean and Std   
acc_mean = np.mean(np.hstack(final_acc_list), axis=1)
acc_std = np.std(np.hstack(final_acc_list), axis=1)

fscore_mean = np.mean(np.hstack(final_fscore_list), axis=1)
fscore_std = np.std(np.hstack(final_fscore_list), axis=1)

kappa_mean, kappa_std =  np.mean(np.hstack(final_kappa_list), axis=1), np.std(np.hstack(final_kappa_list), axis=1)
    
#############################################################    
result_name = 'brazil_fscore'

if result_name == 'brazil_acc':
    _mean_mat = acc_mean.reshape(3,4)   
    _std_mat = acc_std.reshape(3,4)       
    ylabel_name = 'Accuracy'
elif result_name == 'brazil_fscore':
    _mean_mat = fscore_mean.reshape(3,4)  
    _std_mat = fscore_std.reshape(3,4)      
    ylabel_name = 'F1 score'
elif result_name == 'brazil_kappa':
    _mean_mat = kappa_mean.reshape(3,4)     
    _std_mat = kappa_std.reshape(3,4)     
    ylabel_name = 'Kappa'
else:
    print('error')    


#-------#
n_cnn, n_loss = _mean_mat.shape
x = np.arange(0,n_loss) 
y1_mean, y2_mean, y3_mean = _mean_mat[0,:], _mean_mat[1,:], _mean_mat[2,:]
y1_std, y2_std, y3_std = _std_mat[0,:], _std_mat[1,:], _std_mat[2,:]


pl.plot(x, y1_mean, 'D-', color='#CC4F1B', label='1D-2D-CNN', linewidth=2, markersize=8)
pl.fill_between(x, y1_mean-y1_std, y1_mean+y1_std,
    alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')

pl.plot(x, y2_mean, 's-', color='#1B2ACC', label='1D-CNN', linewidth=2, markersize=8)
pl.fill_between(x, y2_mean-y2_std, y2_mean+y2_std,
    alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF')

pl.plot(x, y3_mean, 'o-', color='#3F7F4C', label='2D-CNN', linewidth=2, markersize=8)
pl.fill_between(x, y3_mean-y3_std, y3_mean+y3_std,
    alpha=0.5, edgecolor='#3F7F4C', facecolor='#7EFF99')

loss_label = ['Focal loss', 'CE loss', 'Hybrid loss', 'WCE loss']
plt.xticks(x, loss_label, rotation=0)
plt.legend()

font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 15,
}

plt.xlabel('Loss Function', font2)
plt.ylabel(ylabel_name, font2)

plt.ylim(0.8, 0.95)
# plt.title('Masked and NaN data')
# plt.show()   
   
# save_folder = './plot/'
# bf.make_sure_path_exists(save_folder)  
# plt.savefig(save_folder + result_name)






