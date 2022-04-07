# -*- coding: utf-8 -*-
"""
Created on Fri May  3 07:16:45 2019

@author: Jie
"""

from __future__ import print_function
import numpy as np
import os
import pandas as pd
import librosa
import tensorflow as tf

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

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

from keras import backend as K

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
def feature_extraction(_base_folder, _percent_value, _select_win_len, _select_win_over):
    
    tmp_folder = 'result_all_percent_' + str(_percent_value) + '_winsize_' + str(_select_win_len) + '_winover_' + str(_select_win_over)
    data_folder = os.path.join(_base_folder, tmp_folder)    
    data_list = os.listdir(data_folder)

    # initilization      
    for iData in range(2):
        print(iData)
        
        train_path = os.path.join(data_folder, data_list[iData])
        train_data = pd.read_csv(train_path, low_memory=False, header=None)
        train_data = train_data.values
        n_sample, n_feat = train_data.shape 
        
        train_feature = train_data[:,0:-1]
        train_feat_list = []                
        winsize = select_win_len / 20 
        winstep = winsize / 2                                     
        for i_sample in range(n_sample):    
            
            signal = np.asfortranarray(train_feature[i_sample, :])
            # mel-spectrogram                         
            S = librosa.feature.melspectrogram(y=signal, sr=fs, n_mels=40, n_fft=2048,
                                hop_length=int(winstep), win_length=int(winsize),
                                fmax=fs/2)                   
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            train_feat_list.append(S_dB.ravel()) # label = iList
         
        train_feat_mat = np.asmatrix(train_feat_list)  
        train_label = train_data[:,-1] 
        train_label = train_label.reshape(-1,1)
            
        data_mat = np.hstack((train_feat_mat, train_label))
        
        if iData == 0:
            train_data_mat = data_mat
        elif iData == 1:
            test_data_mat = data_mat
        
    return train_data_mat, test_data_mat, S
        
    
def feat_normalization(training_feat, validation_feat, test_feat):
    
    print('normalization start')
    std_scaler = StandardScaler(copy=False).fit(training_feat)
    training_feat = std_scaler.transform(training_feat)
    validation_feat = std_scaler.transform(validation_feat)
    test_feat = std_scaler.transform(test_feat)
    print('normalization done')    
    
    return training_feat, validation_feat, test_feature


def vector_to_matrix(data_feat, data_label_org, num_classes, S):
    row, col = S.shape
    n_samples_training, n_data = data_feat.shape 
    data_feat_final = []
    for i_samples in range(n_samples_training):
        
        temp_vector = data_feat[i_samples,:]    
        temp_img = temp_vector.reshape(row,col)    
        data_feat_final.append(temp_img[:,:,np.newaxis])

    data_feat_final = np.array(data_feat_final) 
    data_label = to_categorical(np.ravel(data_label_org), num_classes)    
    return data_feat_final, data_label


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



if __name__ == '__main__':
    
    base_folder = r'.\Australia-Frog\raw_data_sliding_noise'                
    base_folder = base_folder.replace('\\', '/')

    snr_list = [-20,-15,-10,-5]
    # snr_list = [-5]
    for snr in snr_list:
        print(snr)
        
        snr_folder = os.path.join(base_folder, str(snr))
        
        noise_list = os.listdir(snr_folder) 
        # noise_list = noise_list[2:]       
        for noise in noise_list:
            print(noise)
            
            noise_folder =os.path.join(snr_folder, noise)

            percent_array = [0.8] 
            for idx in range(len(percent_array)):
                select_percent = percent_array[idx]        
                print(select_percent)
        
                # loop winsize and winover
                win_len_array = np.array([0.2]) * fs
                for select_win_len in win_len_array:
                    select_win_len = int(select_win_len)
                                
                    win_over_array = np.array([0.8])
                    for select_win_over in win_over_array:
                        print('test')
                        
                        shape_a, shape_b = int(select_win_len), 1
                    
                        train_data, test_data, S = feature_extraction(noise_folder, select_percent, select_win_len, select_win_over)
                        
                        #===========================#
                        # convert list to matrix
                        train_feat = train_data[:,0:-1]
                        train_label = train_data[:,-1]
                        
                        test_feat = test_data[:,0:-1]
                        test_label = test_data[:,-1]
                        
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
                        
                        #===========================#
                        training_feat, validation_feat, test_feat = feat_normalization(training_feat, validation_feat, test_feat)    
                 
                        #===========================#                    
                        training_feat_final, training_label = vector_to_matrix(training_feat, training_label_org, num_classes, S)
                        validation_feat_final, validation_label = vector_to_matrix(validation_feat, validation_label_org, num_classes, S)
                        test_feature_final, test_label = vector_to_matrix(test_feat, testLabel, num_classes, S)
                                        
                        #===========================#
                        # loss function
                        loss_fun_array = ['w_cross_entropy']          
                        # loss_fun_array = ['my_cross_entropy','focal', 'twin_loss', 'w_cross_entropy']          
                        nLoss = len(loss_fun_array)
                        for iLoss in range(nLoss):   
                            tmp_loss =  loss_fun_array[iLoss]
                             
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
                                my_loss_fun = my_loss_selection.select_loss(tmp_loss, training_label_org, num_classes, 
                                                                            alpha_value, gamma_value)
                
                            #===========================#
                            # build 2D CNN
                            model = dm_frog.build_2D_CNN_model(training_feat_final, num_classes)
                                                                                    
                            # initiate Adam optimizer
                            opt = keras.optimizers.Adam(lr=0.001)
                                                      
                            model.compile(loss=my_loss_fun,
                                          optimizer=opt,
                                          metrics=['accuracy'])
                            
                            '''
                            saves the model weights after each epoch if the validation loss decreased
                            '''
                            # checkpoint
                            save_weight_folder = os.path.join('./tmp_0421/2D_Aus_org_noise/',tmp_loss)
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
                                            
                            #print('Not using data augmentation.')
                            callbacks_list = [mc, es]
                                                         
                            # Fit the model
                            history = model.fit(training_feat_final, training_label,
                                      batch_size=batch_size,
                                      epochs=epochs,
                                      validation_data=(validation_feat_final, validation_label),
                                      callbacks=callbacks_list,
                                      verbose=0)                
                                            
                            # load the best model and do the classification
                            print('start loading best weights')
                            model = dm_frog.build_2D_CNN_model(training_feat_final, num_classes)
                            
                            # load the model
                            print("Created model and loaded weights from file")
                            model.load_weights(filepath)
                            
                            # Compile model (required to make predictions)
                            model.compile(loss=my_loss_fun, 
                                          optimizer=opt, 
                                          metrics=['accuracy'])
                            
                            predict_label_test = model.predict_classes(test_feature_final)        
                            out_label_prob = model.predict_proba(test_feature_final) 
                        
                            # save classification results   
                            save_folder = './0421/'             
                            save_folder_final = save_folder + '/2D_Aus_org_noise/'+ noise + '_' + str(snr) + '/' + \
                                label_type + '_' + tmp_loss + \
                                '_clean/result_all_percent_' + str(select_percent) + '_winsize_' + \
                                str(select_win_len) + '_winover_' + str(select_win_over) 
                                                                   
                            bf.make_sure_path_exists(save_folder_final)
                            
                            pe.save_csv_performance(testLabel, predict_label_test, out_label_prob, save_folder_final)        
                            pe.loss_acc(history, save_folder_final)      
                        
                        
            
  
