# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 19:16:55 2021

@author: Administrator
"""

import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

#--duration
train_duration_list = []

train_folder = r'.\Brazil-Frog\data_split\training_0.8'
train_audio_list = os.listdir(train_folder)
for train_audio_name in train_audio_list:
    train_audio_path = os.path.join(train_folder, train_audio_name)
    
    y, sr = librosa.load(train_audio_path, sr=None)
    
    train_duration_list.append(len(y) / sr)
    
# print(train_duration_list)
print('Total duration of Train data:' + str(np.sum(train_duration_list)))

test_duration_list = []
test_folder = r'.\Brazil-Frog\data_split\testing_0.8' 
test_audio_list = os.listdir(test_folder)
for test_audio_name in test_audio_list:
    test_audio_path = os.path.join(test_folder, test_audio_name)
    
    y, sr = librosa.load(test_audio_path, sr=None)
    
    test_duration_list.append(len(y) / sr)
    
# print(train_duration_list)
print('Total duration of Test data:' + str(np.sum(test_duration_list)))


#--duration of each species
duration_list = []
all_folder = r'.\Brazil-Frog\data'
all_audio_list = os.listdir(all_folder)
for all_audio_name in all_audio_list:
    all_audio_path = os.path.join(all_folder, all_audio_name)
    
    y, sr = librosa.load(all_audio_path, sr=None)
    
    duration_list.append(len(y) / sr)

AdenomeraAndre_1 = np.sum(duration_list[0:8])
Ameeregatrivittata_1 = np.sum(duration_list[8:13])
hylaedactylus_1 = np.sum(duration_list[13:24])
HylaMinuta_1 = np.sum(duration_list[24:35])
HypsiboasCinerascens_1 = np.sum(duration_list[35:39])
HypsiboasCordobae_1 = np.sum(duration_list[39:43])
LeptodactylusFuscus_1 = np.sum(duration_list[43:47])
OsteocephalusOophagus_1 = np.sum(duration_list[47:50])
Rhinellagranulosa_1 = np.sum(duration_list[50:55])
ScinaxRuber_1 = np.sum(duration_list[55:])


comb_duration = [AdenomeraAndre_1, Ameeregatrivittata_1, hylaedactylus_1, 
                 HylaMinuta_1, HypsiboasCinerascens_1, HypsiboasCordobae_1,
                 LeptodactylusFuscus_1, OsteocephalusOophagus_1, Rhinellagranulosa_1, 
                 ScinaxRuber_1]
# labels = ['Adenomera Andre', 'Ameerega Trivittata', 'Hyla Edactylus', 
#           'Hyla Minuta', 'Hypsiboas Cinerascens', 'Hypsiboas Cordobae',
#           'Leptodactylus Fuscus', 'Osteocephalus Oophagus', 'Rhinella Granulosa',
#           'Scinax Ruber']

labels = ['A.Andre', 'A.Trivittata', 'H.Edactylus', 
          'Hyla Minuta', 'H.Cinerascens', 'H.Cordobae',
          'L.Fuscus', 'O.Oophagus', 'R.Granulosa',
          'S.Ruber']

x = np.arange(len(comb_duration))  # the label locations
width = 0.5  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x, comb_duration, width, label='Men', facecolor='c', edgecolor='b')
plt.ylabel('Duration (s)', fontsize=12)
plt.xlabel('Frog Recording', fontsize=12)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)

# ax.bar_label(rects1, padding=3)
# ax.bar_label(np.array([8,5,11,11,4,4,4,3,5,4]), padding=3)

plt.grid(True)
fig.tight_layout()









