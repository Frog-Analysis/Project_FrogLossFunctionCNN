# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 14:42:35 2020

@author: arnou
"""

# from tensorflow.compat.v1.keras.models import Sequential
from tensorflow.compat.v1.keras.layers import Dense, Activation, Flatten, Dropout, MaxPooling1D, Conv1D, Concatenate
from tensorflow.compat.v1.keras.layers import CuDNNLSTM as LSTM
from tensorflow.compat.v1.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, ZeroPadding2D
# from keras.layers import Conv3D, MaxPooling3D, TimeDistributed
from tensorflow.compat.v1.keras.layers.normalization import BatchNormalization


def build_baseline_model(input1, num_classes):
    
    input_rd = input1

    input_rd = Conv2D(32, (7, 7), padding='same')(input_rd)
    input_rd = BatchNormalization()(input_rd)
    #input_rd = ZeroPadding2D((2, 2))(input_rd)
    input_rd = Activation('relu')(input_rd)
    input_rd = MaxPooling2D(pool_size=(5, 5))(input_rd)
    input_rd = Dropout(0.3)(input_rd)    
    
    input_rd = Conv2D(64, (7, 7), padding='same')(input_rd)
    input_rd = BatchNormalization()(input_rd)
    #input_rd = ZeroPadding2D((2, 2))(input_rd)    
    input_rd = Activation('relu')(input_rd)
    input_rd = MaxPooling2D(pool_size=(4, 100))(input_rd)
    input_rd = Dropout(0.3)(input_rd)    

    out = Flatten()(input_rd)        
    out = Dense(100, activation='relu')(out)
    out = Dropout(0.3)(out)    
    
    out = Dense(num_classes, activation='softmax')(out)

    return out
    


def build_1D_CNN_model(input1, num_classes):

    # build 1D CNN
    input_ft = input1
    input_ft = Conv1D(filters=16, kernel_size=32, strides=2)(input_ft)
    input_ft = BatchNormalization()(input_ft)
    input_ft = Activation("relu")(input_ft)
    input_ft = MaxPooling1D(pool_size=2)(input_ft)
    
    input_ft = Conv1D(filters=32, kernel_size=16, strides=2)(input_ft)
    input_ft = BatchNormalization()(input_ft)
    input_ft = Activation("relu")(input_ft)
    input_ft = MaxPooling1D(pool_size=2)(input_ft)    
    
    input_ft = Conv1D(filters=64, kernel_size=8, strides=2)(input_ft)
    input_ft = BatchNormalization()(input_ft)
    input_ft = Activation("relu")(input_ft)
    input_ft = MaxPooling1D(pool_size=2)(input_ft)
    
    input_ft = LSTM(128, return_sequences=True)(input_ft)
    
    out = Flatten()(input_ft)
    
    out = Dense(100, activation='relu')(out)
    out = Dropout(0.5)(out)
    
    out = Dense(num_classes, activation='softmax')(out)

    return out

 
def build_1D_2D_CNN_model(input1, input2, num_classes):

    # build 1D CNN
    input_ft = input1
    input_ft = Conv1D(filters=16, kernel_size=32, strides=2)(input_ft)
    input_ft = BatchNormalization()(input_ft)
    input_ft = Activation("relu")(input_ft)
    input_ft = MaxPooling1D(pool_size=2)(input_ft)
    
    input_ft = Conv1D(filters=32, kernel_size=16, strides=2)(input_ft)
    input_ft = BatchNormalization()(input_ft)
    input_ft = Activation("relu")(input_ft)
    input_ft = MaxPooling1D(pool_size=2)(input_ft)    
    
    input_ft = Conv1D(filters=64, kernel_size=8, strides=2)(input_ft)
    input_ft = BatchNormalization()(input_ft)
    input_ft = Activation("relu")(input_ft)
    input_ft = MaxPooling1D(pool_size=2)(input_ft)
    
    input_ft = LSTM(128, return_sequences=True)(input_ft)
    input_ft = Flatten()(input_ft)
    
    #------
    input_rd = input2
    input_rd = Conv2D(32, (3, 3))(input_rd)
    input_rd = Activation('relu')(input_rd)
    input_rd = BatchNormalization()(input_rd)
    input_rd = ZeroPadding2D((1, 1))(input_rd)
    input_rd = Conv2D(32, (3, 3))(input_rd)
    input_rd = Activation('relu')(input_rd)
    input_rd = BatchNormalization()(input_rd)
    input_rd = ZeroPadding2D((1, 1))(input_rd)
    input_rd = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_rd)
    input_rd = Dropout(0.2)(input_rd)

    input_rd = Conv2D(64, (3, 3))(input_rd)
    input_rd = Activation('relu')(input_rd)
    input_rd = BatchNormalization()(input_rd)
    input_rd = ZeroPadding2D((1, 1))(input_rd)
    input_rd = Conv2D(64, (3, 3))(input_rd)
    input_rd = Activation('relu')(input_rd)
    input_rd = BatchNormalization()(input_rd)
    input_rd = ZeroPadding2D((1, 1))(input_rd)
    input_rd = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_rd)
    input_rd = Dropout(0.2)(input_rd)    
    
    input_rd = Conv2D(128, (3, 3))(input_rd)
    input_rd = Activation('relu')(input_rd)
    input_rd = BatchNormalization()(input_rd)
    input_rd = ZeroPadding2D((1, 1))(input_rd)
    input_rd = Conv2D(128, (3, 3))(input_rd)
    input_rd = Activation('relu')(input_rd)
    input_rd = BatchNormalization()(input_rd)
    input_rd = ZeroPadding2D((1, 1))(input_rd)
    input_rd = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_rd)
    input_rd = Dropout(0.2)(input_rd)    
    
    input_rd = Flatten()(input_rd)
    
    #-----
    out = Concatenate()([input_ft, input_rd])
    
    out = Dense(100, activation='relu')(out)
    out = Dropout(0.5)(out)
    
    out = Dense(num_classes, activation='softmax')(out)

    return out



def build_2D_CNN_model_FC(input2, num_classes):

    # build 2D CNN
    input_rd = input2
    input_rd = Conv2D(32, (3, 3))(input_rd)
    input_rd = Activation('relu')(input_rd)
    input_rd = BatchNormalization()(input_rd)
    input_rd = ZeroPadding2D((1, 1))(input_rd)
    input_rd = Conv2D(32, (3, 3))(input_rd)
    input_rd = Activation('relu')(input_rd)
    input_rd = BatchNormalization()(input_rd)
    input_rd = ZeroPadding2D((1, 1))(input_rd)
    input_rd = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_rd)
    input_rd = Dropout(0.2)(input_rd)

    input_rd = Conv2D(64, (3, 3))(input_rd)
    input_rd = Activation('relu')(input_rd)
    input_rd = BatchNormalization()(input_rd)
    input_rd = ZeroPadding2D((1, 1))(input_rd)
    input_rd = Conv2D(64, (3, 3))(input_rd)
    input_rd = Activation('relu')(input_rd)
    input_rd = BatchNormalization()(input_rd)
    input_rd = ZeroPadding2D((1, 1))(input_rd)
    input_rd = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_rd)
    input_rd = Dropout(0.2)(input_rd)    
    
    input_rd = Conv2D(128, (3, 3))(input_rd)
    input_rd = Activation('relu')(input_rd)
    input_rd = BatchNormalization()(input_rd)
    input_rd = ZeroPadding2D((1, 1))(input_rd)
    input_rd = Conv2D(128, (3, 3))(input_rd)
    input_rd = Activation('relu')(input_rd)
    input_rd = BatchNormalization()(input_rd)
    input_rd = ZeroPadding2D((1, 1))(input_rd)
    input_rd = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_rd)
    input_rd = Dropout(0.2)(input_rd)    
    
    out = Flatten()(input_rd)
        
    #-----   
    out = Dense(100, activation='relu')(out)
    out = Dropout(0.5)(out)
        
    out = Dense(num_classes, activation='softmax')(out)

    return out


def build_2D_CNN_model_GP(input2, num_classes):

    # build 2D CNN
    input_rd = input2
    input_rd = Conv2D(32, (3, 3))(input_rd)
    input_rd = Activation('relu')(input_rd)
    input_rd = BatchNormalization()(input_rd)
    input_rd = ZeroPadding2D((1, 1))(input_rd)
    input_rd = Conv2D(32, (3, 3))(input_rd)
    input_rd = Activation('relu')(input_rd)
    input_rd = BatchNormalization()(input_rd)
    input_rd = ZeroPadding2D((1, 1))(input_rd)
    input_rd = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_rd)
    input_rd = Dropout(0.2)(input_rd)

    input_rd = Conv2D(64, (3, 3))(input_rd)
    input_rd = Activation('relu')(input_rd)
    input_rd = BatchNormalization()(input_rd)
    input_rd = ZeroPadding2D((1, 1))(input_rd)
    input_rd = Conv2D(64, (3, 3))(input_rd)
    input_rd = Activation('relu')(input_rd)
    input_rd = BatchNormalization()(input_rd)
    input_rd = ZeroPadding2D((1, 1))(input_rd)
    input_rd = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_rd)
    input_rd = Dropout(0.2)(input_rd)    
    
    input_rd = Conv2D(128, (3, 3))(input_rd)
    input_rd = Activation('relu')(input_rd)
    input_rd = BatchNormalization()(input_rd)
    input_rd = ZeroPadding2D((1, 1))(input_rd)
    input_rd = Conv2D(128, (3, 3))(input_rd)
    input_rd = Activation('relu')(input_rd)
    input_rd = BatchNormalization()(input_rd)
    input_rd = ZeroPadding2D((1, 1))(input_rd)
    input_rd = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_rd)
    input_rd = Dropout(0.2)(input_rd)    
    
    out = GlobalAveragePooling2D()(input_rd)
    out = Dropout(0.2)(out)
    out = Dense(100, activation='relu')(out)
    out = Dropout(0.2)(out)
    out = Dense(num_classes, activation='softmax')(out)


    return out



def build_2D_CNN_model_VGG_GP_multi(input1, input2, input3, input4, num_classes):

    # build 2D CNN
    input_a = input1
    input_a = Conv2D(32, (3, 3))(input_a)
    input_a = Activation('relu')(input_a)
    input_a = BatchNormalization()(input_a)
    input_a = ZeroPadding2D((1, 1))(input_a)
    input_a = Conv2D(32, (3, 3))(input_a)
    input_a = Activation('relu')(input_a)
    input_a = BatchNormalization()(input_a)
    input_a = ZeroPadding2D((1, 1))(input_a)
    input_a = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_a)
    input_a = Dropout(0.3)(input_a)

    input_a = Conv2D(64, (3, 3))(input_a)
    input_a = Activation('relu')(input_a)
    input_a = BatchNormalization()(input_a)
    input_a = ZeroPadding2D((1, 1))(input_a)
    input_a = Conv2D(64, (3, 3))(input_a)
    input_a = Activation('relu')(input_a)
    input_a = BatchNormalization()(input_a)
    input_a = ZeroPadding2D((1, 1))(input_a)
    input_a = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_a)
    input_a = Dropout(0.3)(input_a)    
    input_a = Flatten()(input_a)
    
    #--
    input_b = input2
    input_b = Conv2D(32, (3, 3))(input_b)
    input_b = Activation('relu')(input_b)
    input_b = BatchNormalization()(input_b)
    input_b = ZeroPadding2D((1, 1))(input_b)
    input_b = Conv2D(32, (3, 3))(input_b)
    input_b = Activation('relu')(input_b)
    input_b = BatchNormalization()(input_b)
    input_b = ZeroPadding2D((1, 1))(input_b)
    input_b = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_b)
    input_b = Dropout(0.3)(input_b)

    input_b = Conv2D(64, (3, 3))(input_b)
    input_b = Activation('relu')(input_b)
    input_b = BatchNormalization()(input_b)
    input_b = ZeroPadding2D((1, 1))(input_b)
    input_b = Conv2D(64, (3, 3))(input_b)
    input_b = Activation('relu')(input_b)
    input_b = BatchNormalization()(input_b)
    input_b = ZeroPadding2D((1, 1))(input_b)
    input_b = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_b)
    input_b = Dropout(0.3)(input_b)    
    input_b = Flatten()(input_b)


    #--
    input_c = input3
    input_c = Conv2D(32, (3, 3))(input_c)
    input_c = Activation('relu')(input_c)
    input_c = BatchNormalization()(input_c)
    input_c = ZeroPadding2D((1, 1))(input_c)
    input_c = Conv2D(32, (3, 3))(input_c)
    input_c = Activation('relu')(input_c)
    input_c = BatchNormalization()(input_c)
    input_c = ZeroPadding2D((1, 1))(input_c)
    input_c = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_c)
    input_c = Dropout(0.3)(input_c)

    input_c = Conv2D(64, (3, 3))(input_c)
    input_c = Activation('relu')(input_c)
    input_c = BatchNormalization()(input_c)
    input_c = ZeroPadding2D((1, 1))(input_c)
    input_c = Conv2D(64, (3, 3))(input_c)
    input_c = Activation('relu')(input_c)
    input_c = BatchNormalization()(input_c)
    input_c = ZeroPadding2D((1, 1))(input_c)
    input_c = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_c)
    input_c = Dropout(0.3)(input_c)    
    input_c = Flatten()(input_c)

    #--
    input_d = input4
    input_d = Conv2D(32, (3, 3))(input_d)
    input_d = Activation('relu')(input_d)
    input_d = BatchNormalization()(input_d)
    input_d = ZeroPadding2D((1, 1))(input_d)
    input_d = Conv2D(32, (3, 3))(input_d)
    input_d = Activation('relu')(input_d)
    input_d = BatchNormalization()(input_d)
    input_d = ZeroPadding2D((1, 1))(input_d)
    input_d = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_d)
    input_d = Dropout(0.3)(input_d)

    input_d = Conv2D(64, (3, 3))(input_d)
    input_d = Activation('relu')(input_d)
    input_d = BatchNormalization()(input_d)
    input_d = ZeroPadding2D((1, 1))(input_d)
    input_d = Conv2D(64, (3, 3))(input_d)
    input_d = Activation('relu')(input_d)
    input_d = BatchNormalization()(input_d)
    input_d = ZeroPadding2D((1, 1))(input_d)
    input_d = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_d)
    input_d = Dropout(0.3)(input_d)    
    input_d = Flatten()(input_d)

    #--
    out = Concatenate()([input_a, input_b, input_c, input_d])    
    out = Dropout(0.3)(out)
    out = Dense(32, activation='relu')(out)
    out = Dropout(0.3)(out)
    out = Dense(num_classes, activation='softmax')(out)

    return out


def build_2D_CNN_model_baseline_multi(input1, input2, input3, input4, num_classes):

    # build 2D CNN
    input_rd = input1

    input_rd = Conv2D(32, (5, 5))(input_rd)
    input_rd = BatchNormalization()(input_rd)
    input_rd = Activation('relu')(input_rd)
    input_rd = MaxPooling2D(pool_size=(5, 5))(input_rd)
    input_rd = Dropout(0.3)(input_rd)    
    
    input_rd = Conv2D(64, (5, 5))(input_rd)
    input_rd = BatchNormalization()(input_rd)
    input_rd = Activation('relu')(input_rd)
    input_rd = MaxPooling2D(pool_size=(4, 100))(input_rd)
    input_rd = Dropout(0.3)(input_rd)    
    input_a = Flatten()(input_rd)

    #--
    input_rd = input2

    input_rd = Conv2D(32, (5, 5))(input_rd)
    input_rd = BatchNormalization()(input_rd)
    input_rd = Activation('relu')(input_rd)
    input_rd = MaxPooling2D(pool_size=(5, 5))(input_rd)
    input_rd = Dropout(0.3)(input_rd)    
    
    input_rd = Conv2D(64, (5, 5))(input_rd)
    input_rd = BatchNormalization()(input_rd)
    input_rd = Activation('relu')(input_rd)
    input_rd = MaxPooling2D(pool_size=(4, 100))(input_rd)
    input_rd = Dropout(0.3)(input_rd)  
    input_b = Flatten()(input_rd)
    
    #--
    input_rd = input3

    input_rd = Conv2D(32, (5, 5))(input_rd)
    input_rd = BatchNormalization()(input_rd)
    input_rd = Activation('relu')(input_rd)
    input_rd = MaxPooling2D(pool_size=(5, 5))(input_rd)
    input_rd = Dropout(0.3)(input_rd)    
    
    input_rd = Conv2D(64, (5, 5))(input_rd)
    input_rd = BatchNormalization()(input_rd)
    input_rd = Activation('relu')(input_rd)
    input_rd = MaxPooling2D(pool_size=(4, 100))(input_rd)
    input_rd = Dropout(0.3)(input_rd)  
    input_c = Flatten()(input_rd)
    
    #--
    input_rd = input4

    input_rd = Conv2D(32, (5, 5))(input_rd)
    input_rd = BatchNormalization()(input_rd)
    input_rd = Activation('relu')(input_rd)
    input_rd = MaxPooling2D(pool_size=(5, 5))(input_rd)
    input_rd = Dropout(0.3)(input_rd)    
    
    input_rd = Conv2D(64, (5, 5))(input_rd)
    input_rd = BatchNormalization()(input_rd)
    input_rd = Activation('relu')(input_rd)
    input_rd = MaxPooling2D(pool_size=(4, 100))(input_rd)
    input_rd = Dropout(0.3)(input_rd)      
    input_d = Flatten()(input_rd)

    #--
    out = Concatenate()([input_a, input_b, input_c, input_d])        
    out = Dense(100, activation='relu')(out)
    out = Dropout(0.3)(out)    
    
    out = Dense(num_classes, activation='softmax')(out)
    
    return out

