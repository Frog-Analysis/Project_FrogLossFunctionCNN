# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 20:31:02 2021

@author: Administrator
"""

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, MaxPooling1D, Conv1D, Concatenate
from tensorflow.compat.v1.keras.layers import CuDNNLSTM as LSTM
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, ZeroPadding2D
# from keras.layers import Conv3D, MaxPooling3D, TimeDistributed
from keras.layers.normalization import BatchNormalization

#------------------------------------------------------------------------------#
def build_1D_CNN_model(feat_final, num_classes):
    
    # build 1D CNN
    #------#
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=32, strides=2, input_shape = feat_final.shape[1:]))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    
    model.add(Conv1D(filters=32, kernel_size=16, strides=2))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv1D(filters=64, kernel_size=8, strides=2))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    
    model.add(LSTM(128, return_sequences=True))
    # model.add(CuDNNLSTM(128, return_sequences=True))
    
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.summary()

    return model


def build_1D_CNN_model_GAP(feat_final, num_classes):
    
    # build 1D CNN
    #------#
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=32, strides=2, input_shape = feat_final.shape[1:]))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    
    model.add(Conv1D(filters=32, kernel_size=16, strides=2))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Conv1D(filters=64, kernel_size=8, strides=2))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    
    model.add(LSTM(128, return_sequences=True))
    # model.add(CuDNNLSTM(128, return_sequences=True))
    
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.summary()

    return model



def build_2D_CNN_model(feat_final, num_classes):
    
   #------#    
   # build 2D CNN    
   model = Sequential()   
   model.add(Conv2D(32, (3, 3), input_shape = feat_final.shape[1:]))    
   model.add(Activation('relu'))
   model.add(BatchNormalization())
   model.add(ZeroPadding2D((1, 1)))
   model.add(Conv2D(32, (3, 3)))
   model.add(Activation('relu'))
   model.add(BatchNormalization())
   model.add(ZeroPadding2D((1, 1)))
   model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
   model.add(Dropout(0.2))

   model.add(Conv2D(64, (3, 3)))
   model.add(Activation('relu'))
   model.add(BatchNormalization())
   model.add(ZeroPadding2D((1, 1)))
   model.add(Conv2D(64, (3, 3)))
   model.add(Activation('relu'))
   model.add(BatchNormalization())
   model.add(ZeroPadding2D((1, 1)))
   model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
   model.add(Dropout(0.2))

   model.add(Conv2D(128, (3, 3)))
   model.add(Activation('relu'))
   model.add(BatchNormalization())
   model.add(ZeroPadding2D((1, 1)))
   model.add(Conv2D(128, (3, 3)))
   model.add(Activation('relu'))
   model.add(BatchNormalization())
   model.add(ZeroPadding2D((1, 1)))
   model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
   model.add(Dropout(0.2))

   model.add(GlobalAveragePooling2D()) 
   model.add(Dropout(0.2)) # add
   
   model.add(Dense(1000))
   model.add(Activation('relu'))
   model.add(Dropout(0.5))
   
   model.add(Dense(num_classes))
   model.add(Activation('softmax'))
       
   return model   
    


def build_1D_2D_CNN_model(input1, input2, num_classes):
    
    #------#
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
    # input_ft = CuDNNLSTM(128, return_sequences=True)(input_ft)

    input_ft = Flatten()(input_ft)
    
    #------#
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
    
    input_rd = Flatten()(input_rd)
    
    #-----
    out = Concatenate()([input_ft, input_rd])    
    out = Dense(1000, activation='relu')(out)
    out = Dropout(0.5)(out)    
    out = Dense(num_classes, activation='softmax')(out)

    return out






