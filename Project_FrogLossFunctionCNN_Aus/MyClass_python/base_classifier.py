# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 17:56:50 2020

@author: arnou
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier

#import dask
#from dask.distributed import Client
#from sklearn.externals.joblib import parallel_backend
#
#import dask_ml.joblib
#import dask_searchcv as dcv


def radius_knn_classifier(train_feat, train_label):
    knn_param_grid={'radius': [1,5,9],
                    'p': [1, 2]}
    
    #kd_tree_grid = KDTree(train_feat, leaf_size=40, metric='minkowski')
        
    grid = GridSearchCV(RadiusNeighborsClassifier(algorithm='kd_tree'), 
                        param_grid=knn_param_grid, 
                        cv=2, 
                        verbose=1,
                        n_jobs=-1, 
                        scoring='accuracy')
    grid.fit(train_feat, train_label)
    
    return grid



def knn_classifer(train_feat, train_label):

    #--classification
    #knn_param_grid={'n_neighbors':[1,10,20,30,40],
    #                'p':[1,2]}
    
    knn_param_grid={'n_neighbors':[1,10,20,30,40]}
        
    grid = GridSearchCV(KNeighborsClassifier(), param_grid=knn_param_grid, cv=2, n_jobs=-1)
    grid.fit(train_feat, train_label)
    
    return grid


def rf_classifier(train_feat, train_label):


    tree_param_grid={'n_estimators':[2^1,2^3,2^5,2^7,2^9]}
    
    grid = GridSearchCV(RandomForestClassifier(), param_grid=tree_param_grid, cv=2, n_jobs=-1)
    grid.fit(train_feat, train_label)

    return grid



#def rf_classifier_fast(train_feat, train_label):
#    
#    #client = Client() # start a local Dask client
#
#    with parallel_backend('dask'):
#        
#        param_grid = {
#        'bootstrap': [True],
#        'max_depth': [8, 9],
#        'max_features': [2, 3],
#        'min_samples_leaf': [4, 5],
#        'min_samples_split': [8, 10],
#        'n_estimators': [100, 200]
#        }
#    
#        # Create a based model
#        rf = RandomForestClassifier()
#        
#    grid_search = dcv.GridSearchCV(estimator = rf, param_grid = param_grid, cv = 2)
#    grid_search.fit(train_feat, train_label)
#    
#    return grid_search























