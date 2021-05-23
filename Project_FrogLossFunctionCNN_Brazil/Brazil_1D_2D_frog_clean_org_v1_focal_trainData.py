# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 17:12:34 2021

@author: arnou
"""

from __future__ import print_function
import numpy as np
import os
import pandas as pd
import librosa

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from MyClass_python import base_function as bf
from MyClass_python import plot_evaluation as pe
from MyClass_python import my_loss_selection
from MyClass_python import deep_model_frog_activity as dm_frog
# from MyClass_python import my_loss_function as loss

import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils.np_utils import to_categorical
from keras.models import Input, Model

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

#===========================#
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#===========================#
np.random.seed(1337)
    
#-----------------------------------------------------------------------------#
def vector_to_matrix(data_feat, data_label_org, num_classes):
    n_samples_training, n_data = data_feat.shape 
    data_feat_final = []
    for i_samples in range(n_samples_training):
        
        temp_vector = data_feat[i_samples,:]    
        temp_img = temp_vector.reshape(row,col)    
        data_feat_final.append(temp_img[:,:,np.newaxis])

    data_feat_final = np.array(data_feat_final) 
    data_label = to_categorical(np.ravel(data_label_org), num_classes)    
    return data_feat_final, data_label

#-----------------------------------------------------------------------------#
label_type = 'weak'

#-----------------------------------------------------------------------------#    
num_classes = 11
batch_size = 64
epochs = 200
fs = 44100

#-----------------------------------------------------------------------------#
def feature_extraction(_base_folder, _percent_value, _select_win_len, _select_win_over):
    
    tmp_folder = 'percent_' + str(_percent_value) + '_winsize_' + str(_select_win_len) + '_winover_' + str(_select_win_over)
    species_folder = os.path.join(_base_folder, tmp_folder)    
    species_list = os.listdir(species_folder)
    
    # initilization
    train_data_list, test_data_list = [], []    
    train_data_list_waveform, test_data_list_waveform = [], []
    
    nAudio = len(species_list)
    for iAudio in range(nAudio):    
        print(species_list[iAudio] + ':' + str(nAudio) + ':' + str(iAudio))
        
        audio_folder = os.path.join(species_folder, species_list[iAudio])
        
        #===========#
        train_data_path = os.path.join(audio_folder, 'train.csv')                                
        train_data_single = pd.read_csv(train_data_path, low_memory=False, header=None)                
        train_data_single = train_data_single.values
        train_data_list_waveform.append(train_data_single)
                       
        n_sample, n_feat = train_data_single.shape    
        train_feature = train_data_single[:,0:-1]
        train_feat_list = []                
        winsize = select_win_len / 20 
        winstep = winsize / 2                                     
        for i_sample in range(n_sample):
            # print(i_sample)
            signal = np.asfortranarray(train_feature[i_sample, :])
            # mel-spectrogram                         
            S = librosa.feature.melspectrogram(y=signal, sr=fs, n_mels=40, n_fft=2048, 
                               hop_length=int(winstep), win_length=int(winsize),
                               fmax=fs/2)                   
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            train_feat_list.append(S_dB.ravel()) # label = iList
                
        train_feat_mat = np.asmatrix(train_feat_list)   
        train_label = train_data_single[:,-1]
        train_label = train_label.reshape(-1,1)
                 
        train_data_list.append(np.hstack((train_feat_mat, train_label)))
        
        #===========#
        test_data_path = os.path.join(audio_folder, 'test.csv') 
        test_data_single = pd.read_csv(test_data_path, low_memory=False, header=None)                
        test_data_single = test_data_single.values
        test_data_list_waveform.append(test_data_single)
        
        n_sample, n_feat = test_data_single.shape    
        test_feature = test_data_single[:,0:-1]
        test_feat_list = []                                                       
        for i_sample in range(n_sample):
            # print(i_sample)
            signal = np.asfortranarray(test_feature[i_sample, :])
            # mel-spectrogram                         
            S = librosa.feature.melspectrogram(y=signal, sr=fs, n_mels=40, n_fft=2048, 
                               hop_length=int(winstep), win_length=int(winsize),
                               fmax=fs/2)                   
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            test_feat_list.append(S_dB.ravel()) # label = iList
                
        test_feat_mat = np.asmatrix(test_feat_list)                
        test_label = test_data_single[:,-1]
        test_label = test_label.reshape(-1,1)
        
        test_data_list.append(np.hstack((test_feat_mat, test_label)))
        
        #===========#
                    
    return train_data_list, test_data_list, train_data_list_waveform, test_data_list_waveform, S



if __name__ == '__main__':
    
    base_folder = r'.\Brazil-Frog\0329_raw_data_clean'                
    base_folder = base_folder.replace('\\', '/')

    percent_array = [0.8] 
    # percent_array = [0.5,0.6,0.7,0.8,0.9]   
    for idx in range(len(percent_array)):
        select_percent = percent_array[idx]        
        print(select_percent)

        # loop winsize and winover
        # win_len_array = np.array([0.2, 0.5, 1]) * fs
        win_len_array = np.array([0.2]) * fs
        for select_win_len in win_len_array:
            select_win_len = int(select_win_len)
                        
            win_over_array = np.array([0.8])
            for select_win_over in win_over_array:
    
                train_data_list, test_data_list, train_data_list_waveform, test_data_list_waveform, S = feature_extraction(base_folder, select_percent, select_win_len, select_win_over)
                
                row, col = S.shape
                
                #===========================#
                # convert list to matrix
                train_data = np.vstack(train_data_list)
                train_feat = train_data[:,0:-1]
                train_label = train_data[:,-1]
 
                nRepeat = 5
                for iRepeat in range(nRepeat):                    
                    repeat_name = 'repeat_' + str(iRepeat)
                    print(repeat_name)                

                    test_data = np.vstack(test_data_list)
                    test_feat = test_data[:,0:-1]
                    test_label = test_data[:,-1]
                    
                    sample_index = np.arange(train_label.shape[0])
                    train_index, test_index = train_test_split(sample_index, test_size=0.2)
                    training_index, validation_index = train_test_split(train_index, test_size=0.25)
                                
                    training_feat = train_feat[training_index,:]
                    validation_feat = train_feat[validation_index,:]
                    test_feat = train_feat[test_index,:]
                
                    training_label = train_label[training_index]
                    validation_label = train_label[validation_index]
                    test_label = train_label[test_index]
                    
                    #===========================#
                    training_label_org = np.ravel(training_label)
                    validation_label_org = np.ravel(validation_label)
                    testLabel = np.ravel(test_label)
                    
                    print('normalization')
                    training_feat = StandardScaler(copy=False).fit_transform(training_feat)
                    validation_feat = StandardScaler(copy=False).fit_transform(validation_feat)
                    testFeature = StandardScaler(copy=False).fit_transform(test_feat)
    
                    print('reshape train spectrogram data')
                    training_feat_final, training_label = vector_to_matrix(training_feat, training_label_org, num_classes)
                    validation_feat_final, validation_label = vector_to_matrix(validation_feat, validation_label_org, num_classes)
                    testFeature_final, test_label = vector_to_matrix(testFeature, testLabel, num_classes)
                
                    #===========================#
                    train_data_waveform = np.vstack(train_data_list_waveform)                
                    train_feat_waveform = train_data_waveform[:,0:-1]
                    
                    training_feat_waveform = train_feat_waveform[training_index,:]
                    validation_feat_waveform = train_feat_waveform[validation_index,:]
                    test_feat_waveform = train_feat_waveform[test_index,:]
                    
                    training_feat_waveform = StandardScaler(copy=False).fit_transform(training_feat_waveform)
                    validation_feat_waveform = StandardScaler(copy=False).fit_transform(validation_feat_waveform)
                    testFeature_waveform = StandardScaler(copy=False).fit_transform(test_feat_waveform)
                
                    #===========================#
                    select_win_len, fs = 0.2, 44100
                    
                    shape_a = int(select_win_len*fs)
                    shape_b = 1
                    
                    print('reshape train waveform data')
                    training_feat_final_waveform = training_feat_waveform.reshape(training_feat_waveform.shape[0], shape_a, shape_b).astype('float32')            
                    validation_feat_final_waveform = validation_feat_waveform.reshape(validation_feat_waveform.shape[0], shape_a, shape_b).astype('float32')            
                    testFeature_final_waveform = testFeature_waveform.reshape(testFeature_waveform.shape[0], shape_a, shape_b).astype('float32')            
                
                    #===========================#
                    # loss function
                    alpha_array = np.array([ 0.1, 0.25, 0.5, 0.75])  
                    # alpha_array = np.array([ 0.75])  
                    for alpha_value in alpha_array:
                        
                        gamma_array = np.array([2, 4, 6])
                        # gamma_array = np.array([4, 6])
                        for gamma_value in gamma_array:
                            print(alpha_value, gamma_value)
                            
                            alpha_value = float(alpha_value)
                            gamma_value = int(gamma_value)
            
                            loss_fun_array = ['focal']   
                            nLoss = len(loss_fun_array)
                            for iLoss in range(nLoss):   
                                tmp_loss =  loss_fun_array[iLoss]
                                
                                my_loss_fun = my_loss_selection.select_loss(tmp_loss, training_label_org, num_classes, 
                                                                            alpha_value, gamma_value)    
                          
                                #===========================#
                                # build 1D-2D CNN
                                input_1d = Input(shape=training_feat_final_waveform.shape[1:])
                                input_2d = Input(shape=training_feat_final.shape[1:])
                            
                                output = dm_frog.build_1D_2D_CNN_model(input_1d, input_2d, num_classes)
                                model = Model(inputs=[input_1d, input_2d], outputs=output)                                          
                                              
                                # initiate Adam optimizer
                                opt = keras.optimizers.Adam(lr=0.0001)
                                
                                model.compile(loss=my_loss_fun,
                                              optimizer=opt,
                                              metrics=['accuracy'])
                                
                                '''
                                saves the model weights after each epoch if the validation loss decreased
                                '''
                                # checkpoint
                                save_weight_folder = os.path.join('./tmp0429/1D_2D_Brazil_org/',tmp_loss)
                                bf.make_sure_path_exists(save_weight_folder)
                                filepath= save_weight_folder + "weights.best.org.left.hdf5"   
                                
                                mc = ModelCheckpoint(filepath, 
                                                     monitor='val_accuracy', 
                                                     verbose=1, 
                                                     save_best_only=True, 
                                                     mode='max')
                                
                                es = EarlyStopping(monitor='val_loss', 
                                            mode='min', 
                                            verbose=1, 
                                            patience=20)
                                                
                                #print('NO data augmentation.')
                                callbacks_list = [mc, es]
                                                             
                                # Fit the model
                                history = model.fit([training_feat_final_waveform, training_feat_final], training_label,
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          validation_data=([validation_feat_final_waveform, validation_feat_final], validation_label),
                                          callbacks=callbacks_list,
                                          verbose=0)                
                                                
                                # load the best model and do the classification
                                print('start loading best weights')
                                # build 1D-2D CNN
                                output = dm_frog.build_1D_2D_CNN_model(input_1d, input_2d, num_classes)
                                model = Model(inputs=[input_1d, input_2d], outputs=output)  
                                
                                # load the model
                                print("Created model and loaded weights from file")
                                model.load_weights(filepath)
                                
                                # Compile model (required to make predictions)
                                model.compile(loss=my_loss_fun, 
                                              optimizer=opt, 
                                              metrics=['accuracy'])
                                
                                out_label_prob = model.predict([testFeature_final_waveform, testFeature_final]) 
                                predict_label_test = np.argmax(out_label_prob, axis=1)
                                
                                # save classification results   
                                save_folder = './0429/'             
                                save_folder_final = save_folder + '/1D_2D_Brazil_org_focal/' + repeat_name + '/' + \
                                    label_type + '_' + tmp_loss + \
                                    '_clean_alpha_' + str(alpha_value) + '_gamma_' + str(gamma_value) + \
                                    '/result_all_percent_' + str(select_percent) + '_winsize_' + \
                                    str(select_win_len) + '_winover_' + str(select_win_over)
                                                                       
                                bf.make_sure_path_exists(save_folder_final)
                                
                                pe.save_csv_performance(testLabel, predict_label_test, out_label_prob, save_folder_final)        
                                pe.loss_acc(history, save_folder_final)    
                               
                    





