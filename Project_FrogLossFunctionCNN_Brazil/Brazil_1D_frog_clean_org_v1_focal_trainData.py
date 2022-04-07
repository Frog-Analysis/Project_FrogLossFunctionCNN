# -*- coding: utf-8 -*-
"""
Created on Fri May  3 07:16:45 2019

@author: Jie
"""

from __future__ import print_function
import numpy as np
import os
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

import tensorflow as tf

from MyClass_python import base_function as bf
from MyClass_python import plot_evaluation as pe
from MyClass_python import my_loss_selection, my_loss_function
from MyClass_python import deep_model_frog_activity as dm_frog

import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
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
                
    nSpecies = len(species_list)
    for iSpecies in range(nSpecies):    
        print(iSpecies)
                
        audio_folder = os.path.join(species_folder, species_list[iSpecies])
        
        train_data_path = os.path.join(audio_folder, 'train.csv')                                
        train_data_single = pd.read_csv(train_data_path, low_memory=False, header=None)                
        train_data_list.append(train_data_single.values)
        
        test_data_path = os.path.join(audio_folder, 'test.csv') 
        test_data_single = pd.read_csv(test_data_path, low_memory=False, header=None)                
        test_data_list.append(test_data_single.values)
                    
    return train_data_list, test_data_list


def feat_normalization(training_feat, validation_feat, test_feat):
    
    print('normalization start')
    std_scaler = StandardScaler(copy=False).fit(training_feat)
    training_feat = std_scaler.transform(training_feat)
    validation_feat = std_scaler.transform(validation_feat)
    test_feature = std_scaler.transform(test_feat)  
    print('normalization done')    
    
    return training_feat, validation_feat, test_feature


def feat_reshape(training_feat, validation_feat, test_feature):
    
    print('reshape data start')   
    feat_dict = {'tranining':[], 'validation':[], 'test':[]}
    label_dict = {'tranining':[], 'validation':[], 'test':[]}
    
    # reshape training data
    print('reshape training')
    training_feat_final = training_feat.reshape(training_feat.shape[0], shape_a, shape_b).astype('float32')            
    training_label = keras.utils.to_categorical(np.ravel(training_label_org), num_classes)    
       
    # reshape validation data
    print('reshape validation')
    validation_feat_final = validation_feat.reshape(validation_feat.shape[0], shape_a, shape_b).astype('float32')            
    validation_label = keras.utils.to_categorical(np.ravel(validation_label_org), num_classes)    
     
    # reshape test data
    print('reshape test')
    test_feature_final = test_feature.reshape(test_feature.shape[0], shape_a, shape_b).astype('float32')            
    test_label = keras.utils.to_categorical(np.ravel(testLabel), num_classes)    
    
    feat_dict['training'] = training_feat_final
    feat_dict['validation'] = validation_feat_final
    feat_dict['test'] = test_feature_final
    
    label_dict['training'] = training_label
    label_dict['validation'] = validation_label
    label_dict['test'] = test_label
    print('reshape data start')   

    return feat_dict, label_dict

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
                print('test')
                
                shape_a, shape_b = int(select_win_len), 1            
                train_data_list, test_data_list = feature_extraction(base_folder, select_percent, select_win_len, select_win_over)
                
                #===========================#
                # convert list to matrix
                train_data = np.vstack(train_data_list)
                train_feat = train_data[:,0:-1]
                train_label = train_data[:,-1]

                nRepeat = 5
                for iRepeat in range(nRepeat):                    
                    repeat_name = 'repeat_' + str(iRepeat)
                    print(repeat_name)
                
                    #===========================#
                    sample_index = np.arange(train_label.shape[0])
                    train_index, test_index = train_test_split(sample_index, test_size=0.2)
                    training_index, validation_index = train_test_split(train_index, test_size=0.25)
    
                    #===========================#
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
                                   
                    #===========================#
                    training_feat, validation_feat, test_feat = feat_normalization(training_feat, validation_feat, test_feat)    
             
                    #===========================#                    
                    feat_dict, label_dict = feat_reshape(training_feat, validation_feat, test_feat)
                    
                    training_feat_final = feat_dict['training']  
                    validation_feat_final = feat_dict['validation']  
                    test_feature_final = feat_dict['test']      
                    
                    training_label = label_dict['training'] 
                    validation_label = label_dict['validation'] 
                    # test_label = label_dict['test']     
                    
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
                                
                                my_loss_fun = my_loss_selection.select_loss(tmp_loss, training_label_org, num_classes)    
                                #===========================#
                                # build 1D CNN
                                model = dm_frog.build_1D_CNN_model(training_feat_final, num_classes)
                                                                                        
                                # initiate Adam optimizer
                                opt = keras.optimizers.Adam(lr=0.0001)
                                                          
                                model.compile(loss=my_loss_fun,
                                              optimizer=opt,
                                              metrics=['accuracy'])
                                
                                '''
                                saves the model weights after each epoch if the validation loss decreased
                                '''
                                # checkpoint
                                save_weight_folder = os.path.join('./tmp0429/1D_Brazil_org/',tmp_loss)
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
                                model = dm_frog.build_1D_CNN_model(training_feat_final, num_classes)
                                
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
                                save_folder = './0429/'             
                                save_folder_final = save_folder + '/1D_Brazil_org_focal/' + repeat_name + '/' + \
                                    label_type + '_' + tmp_loss + \
                                    '_clean_alpha_' + str(alpha_value) + '_gamma_' + str(gamma_value) + \
                                    '/result_all_percent_' + str(select_percent) + '_winsize_' + \
                                    str(select_win_len) + '_winover_' + str(select_win_over)
                                                               
                                bf.make_sure_path_exists(save_folder_final)
                                
                                pe.save_csv_performance(testLabel, predict_label_test, out_label_prob, save_folder_final)        
                                pe.loss_acc(history, save_folder_final)      
                
                
            
            
        
  
