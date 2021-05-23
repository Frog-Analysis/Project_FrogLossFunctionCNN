# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 14:31:49 2020

@author: arnoud

% Self-defined Loss Functions

"""

import keras.backend as K
import tensorflow as tf
import numpy as np
#from itertools import product


def focal_loss_fixed(y_true, y_pred, alpha_value, gamma_value):
    """Focal loss for multi-classification
    FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
    Notice: y_pred is probability after softmax
    gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
    d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
    Focal Loss for Dense Object Detection
    https://arxiv.org/abs/1708.02002

    Arguments:
        y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
        y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

    Keyword Arguments:
        gamma {float} -- (default: {2.0})
        alpha {float} -- (default: {4.0})

    Returns:
        [tensor] -- loss.
    """
    
    alpha, gamma = alpha_value, gamma_value

    epsilon = 1.e-9
    y_true = tf.convert_to_tensor(y_true, tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, tf.float32)

    model_out = tf.math.add(y_pred, epsilon)
    ce = tf.math.multiply(y_true, -tf.math.log(model_out))
    weight = tf.math.multiply(y_true, tf.math.pow(tf.subtract(1., model_out), gamma))
    fl = tf.math.multiply(alpha, tf.math.multiply(weight, ce))
    reduced_fl = tf.reduce_max(fl, axis=1)
    return tf.reduce_mean(reduced_fl)


def wcce(y_true, y_pred, weights):
    Kweights = K.constant(weights)
    if not K.is_tensor(y_pred): y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    return K.categorical_crossentropy(y_true, y_pred) * K.sum(y_true * Kweights, axis=-1)


#def w_categorical_crossentropy(y_true, y_pred, weights):
#    nb_cl = len(weights)
#    final_mask = K.zeros_like(y_pred[:, 0])
#    y_pred_max = K.max(y_pred, axis=1)
#    y_pred_max = K.expand_dims(y_pred_max, 1)
#    y_pred_max_mat = K.equal(y_pred, y_pred_max)
#    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
#        final_mask += (K.cast(weights[c_t, c_p],K.floatx()) * K.cast(y_pred_max_mat[:, c_p] ,K.floatx())* K.cast(y_true[:, c_t],K.floatx()))
#    return K.categorical_crossentropy(y_pred, y_true) * final_mask


def my_categorical_crossentropy(y_true, y_pred):    
   loss = K.categorical_crossentropy(y_true,y_pred) 
   return loss
    
                                                                       
def score_loss(y_true, y_pred, n_class):
    loss = 0
    # number of classes
    for i in np.eye(n_class):
        y_true_ = K.constant([list(i)]) * y_true
        y_pred_ = K.constant([list(i)]) * y_pred
        loss += 0.5 * K.sum(y_true_ * y_pred_) / K.sum(y_true_ + y_pred_ + K.epsilon())
    return - K.log(loss + K.epsilon())
    
    

    







