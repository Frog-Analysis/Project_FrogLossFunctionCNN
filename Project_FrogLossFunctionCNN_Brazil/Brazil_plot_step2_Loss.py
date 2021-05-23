# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 19:02:25 2021

@author: Administrator
"""

import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#--figure 1

base_folder = r'.\0408\figure1'

"""
Loss function + CNN architecture

"""

acc_list = []
fscore_list = []
kappa_list = []
method_list = []

cnn_arch_list = os.listdir(base_folder)
for cnn_arch in cnn_arch_list:
    # print(cnn_arch)
    
    cnn_arch_folder = os.path.join(base_folder, cnn_arch)
    
    cnn_loss_list = os.listdir(cnn_arch_folder) 
    cnn_loss_list = cnn_loss_list[0:3] + cnn_loss_list[4:]
    print(cnn_loss_list)
    for cnn_loss in cnn_loss_list:
        
        cnn_loss_path = os.path.join(cnn_arch_folder, cnn_loss)
        
        tmp_list = os.listdir(cnn_loss_path)
        for tmp_name in tmp_list:
            final_path = os.path.join(cnn_loss_path, tmp_name)
            print(final_path)
            
            acf_data = pd.read_csv(os.path.join(final_path, 'accuracy_fscore_kappa.csv'), header=None)
            acf_value = acf_data.values
            
            acc_list.append(acf_value[0])
            fscore_list.append(acf_value[1])
            kappa_list.append(acf_value[2])
            
            method_list.append([cnn_arch, cnn_loss])
            
            
acc_array = np.vstack(acc_list)
fscore_array = np.vstack(fscore_list)
kappa_array = np.vstack(kappa_list)

result_name = 'aus_acc'

if result_name == 'aus_acc':
    acc_mat = acc_array.reshape(3,4)   
    ylabel_name = 'Accuracy'
elif result_name == 'aus_fscore':
    acc_mat = fscore_array.reshape(3,4)   
    ylabel_name = 'F1 score'
else:
    acc_mat = kappa_array.reshape(3,4)     
    ylabel_name = 'Kappa'
    


n_cnn, n_loss = acc_mat.shape
           
x = np.arange(0,n_loss) 
y1 = acc_mat[0,:]
y2 = acc_mat[1,:]
y3 = acc_mat[2,:]

fig, ax = plt.subplots()
       
plt.plot(x, y1, 'o-', color='lightgrey', label='1D-2D-CNN', linewidth=2, markersize=8)
plt.plot(x, y2, 's-', label='1D-CNN', linewidth=2, markersize=8)
plt.plot(x, y3, 'D-', label='2D-CNN', linewidth=2, markersize=8)
plt.legend()

loss_label = ['Focal loss', 'CE loss', 'Hybrid loss', 'WCE loss']
plt.xticks(x, loss_label, rotation=0)
plt.grid('on')

font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 15,
}

plt.xlabel('Loss Function', font2)
plt.ylabel(ylabel_name, font2)
plt.ylim(0.8, 0.92)
# plt.title('Masked and NaN data')
# plt.show()        
plt.savefig('./plot/' + result_name)





