# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:58:22 2020

@author: arnou
"""

# import numpy as np
from MyClass_python import my_loss_function as loss


def select_loss(loss_method, training_label_org, nb_classes, alpha_value=0.25, gamma_value=4):
    
    print(alpha_value, gamma_value)                
    
    if loss_method == 'triple_loss' :
        my_loss = lambda y_true, y_pred: 1 / 3 * loss.focal_loss_fixed(y_true, y_pred, alpha_value, gamma_value) + \
                                         1 / 3 * loss.my_categorical_crossentropy(y_true, y_pred) + \
                                         1 / 3 * loss.score_loss(y_true, y_pred)

    elif loss_method == 'twin_loss' :
        my_loss = lambda y_true, y_pred: 1 / 2 * loss.focal_loss_fixed(y_true, y_pred, alpha_value, gamma_value) + \
                                         1 / 2 * loss.my_categorical_crossentropy(y_true, y_pred)
                                         
    elif loss_method == 'focal' :
        my_loss = lambda y_true, y_pred: loss.focal_loss_fixed(y_true, y_pred, alpha_value, gamma_value)
                    
    elif loss_method == 'my_cross_entropy' :
        my_loss = lambda y_true, y_pred: loss.my_categorical_crossentropy(y_true, y_pred)   

    # elif loss_method == 'w_cross_entropy' :
    #     weights, probablity = np.histogram(training_label_org, bins=nb_classes, range=[0, nb_classes-1])            
    #     class_props = weights / np.sum(weights)
    #     w_array = np.ones((nb_classes,nb_classes))
    #     w_array[np.diag_indices(nb_classes)] = np.log(1/class_props)
    #     #w_array[np.diag_indices(nb_classes)] = np.max(class_props) / class_props
    #     my_loss = lambda y_true, y_pred: loss.w_categorical_crossentropy(y_true, y_pred, weights=w_array)
            
    else:
        print('please input right loss function')
    
    return my_loss
    
    











