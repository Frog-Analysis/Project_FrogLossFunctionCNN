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
import tensorflow as tf

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()  #disable for tensorFlow V2
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from MyClass_python import base_function as bf
from MyClass_python import plot_evaluation as pe
from MyClass_python import my_loss_selection
from MyClass_python import deep_model_frog_activity as dm_frog
from sklearn.utils import class_weight

import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils.np_utils import to_categorical
from keras.models import Input, Model
from keras import backend as K

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

#===========================#
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#===========================#
np.random.seed(1337)
    
#-----------------------------------------------------------------------------#

label_type = 'weak'

#-----------------------------------------------------------------------------#    
num_classes = 24
batch_size = 64
epochs = 200
fs = 44100

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
def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss


def twin_loss_weight(alpha_value, gamma_value, weights):
    
    weights = K.variable(weights)

    def loss(y_true, y_pred):
        
        # weighted CE-loss
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss_w = -K.sum(loss, -1)
                
        #---focal loss
        alpha, gamma = alpha_value, gamma_value

        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)
    
        model_out = tf.math.add(y_pred, epsilon)
        ce = tf.math.multiply(y_true, -tf.math.log(model_out))
        weight = tf.math.multiply(y_true, tf.math.pow(tf.subtract(1., model_out), gamma))
        fl = tf.math.multiply(alpha, tf.math.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        loss_focal = tf.reduce_mean(reduced_fl)
        
        loss = 1/2*loss_w + 1/2*loss_focal
                
        return loss

    return loss

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
    
    base_folder = r'.\Australia-Frog\0409_raw_data_clean'                
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
                
                test_data = np.vstack(test_data_list)
                test_feat = test_data[:,0:-1]
                test_label = test_data[:,-1]

                nRepeat = 5
                for iRepeat in range(nRepeat):                    
                    repeat_name = 'repeat_' + str(iRepeat)
                    print(repeat_name)
                
                    sample_index = np.arange(train_label.shape[0])
                    training_index, validation_index = train_test_split(sample_index, test_size=0.2)
                                
                    training_feat = train_feat[training_index,:]
                    validation_feat = train_feat[validation_index,:]
                
                    training_label = train_label[training_index]
                    validation_label = train_label[validation_index]
                    
                    #===========================#
                    training_label_org = np.ravel(training_label)
                    validation_label_org = np.ravel(validation_label)
                    testLabel = np.ravel(test_label)
                    
                    print('normalization')
                    training_feat = StandardScaler(copy=False).fit_transform(training_feat)
                    validation_feat = StandardScaler(copy=False).fit_transform(validation_feat)
                    testFeature = StandardScaler(copy=False).fit_transform(test_feat)
                
                    #===========================#
                    train_data_waveform = np.vstack(train_data_list_waveform)
                    test_data_waveform = np.vstack(test_data_list_waveform)
                    
                    train_feat_waveform = train_data_waveform[:,0:-1]
                    test_feat_waveform = test_data_waveform[:,0:-1]
                    
                    training_feat_waveform = train_feat_waveform[training_index,:]
                    validation_feat_waveform = train_feat_waveform[validation_index,:]
                    
                    training_feat_waveform = StandardScaler(copy=False).fit_transform(training_feat_waveform)
                    validation_feat_waveform = StandardScaler(copy=False).fit_transform(validation_feat_waveform)
                    testFeature_waveform = StandardScaler(copy=False).fit_transform(test_feat_waveform)
                    
                    #===========================#
                    print('reshape train data')
                    training_feat_final, training_label = vector_to_matrix(training_feat, training_label_org, num_classes)
                    validation_feat_final, validation_label = vector_to_matrix(validation_feat, validation_label_org, num_classes)
                    testFeature_final, test_label_final = vector_to_matrix(testFeature, testLabel, num_classes)
    
                    #===========================#
                    select_win_len, fs = 0.2, 44100
                    
                    shape_a = int(select_win_len*fs)
                    shape_b = 1
                    
                    training_feat_final_waveform = training_feat_waveform.reshape(training_feat_waveform.shape[0], shape_a, shape_b).astype('float32')            
                    validation_feat_final_waveform = validation_feat_waveform.reshape(validation_feat_waveform.shape[0], shape_a, shape_b).astype('float32')            
                    testFeature_final_waveform = testFeature_waveform.reshape(testFeature_waveform.shape[0], shape_a, shape_b).astype('float32')            
                
                    # loss function
                    loss_fun_array = ['my_cross_entropy', 'w_cross_entropy', 'focal', 'twin_loss']
                    # loss_fun_array = ['twin_loss']
                    nLoss = len(loss_fun_array)
                    for iLoss in range(nLoss):   
                        tmp_loss = loss_fun_array[iLoss]
                         
                        #===========================#
                        # build 1D-2D CNN
                        input_1d = Input(shape=training_feat_final_waveform.shape[1:])
                        input_2d = Input(shape=training_feat_final.shape[1:])
                    
                        output = dm_frog.build_1D_2D_CNN_model(input_1d, input_2d, num_classes)
                        model = Model(inputs=[input_1d, input_2d], outputs=output)                                          
                                      
                        # initiate Adam optimizer
                        opt = keras.optimizers.Adam(lr=0.0001)
                        
                        if tmp_loss == 'w_cross_entropy':
                            weights = class_weight.compute_class_weight('balanced', np.unique(training_label_org), 
                                                                        training_label_org) 
                            my_loss_fun = weighted_categorical_crossentropy(weights)                     
                        
                        elif tmp_loss == 'twin_loss_weight':
                            alpha_value, gamma_value = 0.25, 4   
                            weights = class_weight.compute_class_weight('balanced', np.unique(training_label_org), 
                                                                        training_label_org) 
                            my_loss_fun = twin_loss_weight(alpha_value, gamma_value, weights)
                            
                        else:    
                            alpha_value, gamma_value = 0.25, 4       
                            my_loss_fun = my_loss_selection.select_loss(tmp_loss, training_label_org, num_classes, alpha_value, gamma_value)
      
                        #--start training
                        model.compile(loss=my_loss_fun,
                                      optimizer=opt,
                                      metrics=['accuracy'])
                        
                        '''
                        saves the model weights after each epoch if the validation loss decreased
                        '''
                        # checkpoint
                        save_weight_folder = os.path.join('./tmp_0429/1D_2D_Aus_org/',tmp_loss)
                        bf.make_sure_path_exists(save_weight_folder)
                        filepath= save_weight_folder + "/weights.best.org.left.hdf5"   
                        
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
                        save_folder_final = save_folder + '/1D_2D_Aus_org/' + repeat_name + '/' + \
                             label_type + '_' + tmp_loss + \
                            '_clean/result_all_percent_' + str(select_percent) + '_winsize_' + \
                            str(int(select_win_len*fs)) + '_winover_' + str(select_win_over) 
                                                               
                        bf.make_sure_path_exists(save_folder_final)
                        
                        pe.save_csv_performance(testLabel, predict_label_test, out_label_prob, save_folder_final)        
                        pe.loss_acc(history, save_folder_final)    
                   
        





