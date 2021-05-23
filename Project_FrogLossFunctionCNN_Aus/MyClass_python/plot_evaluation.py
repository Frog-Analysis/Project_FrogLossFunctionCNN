# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 14:38:06 2020

@author: arnou
"""

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import numpy as np
from itertools import cycle
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score
from sklearn.metrics import confusion_matrix

def draw_POC(Y_valid, Y_pred, fold_pic, num_classes, method):
    
    # roc_curve:真正率（True Positive Rate , TPR）或灵敏度（sensitivity）
    # 横坐标：假正率（False Positive Rate , FPR）
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_valid[:, i], Y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_valid.ravel(), Y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                    
    # Compute macro-average ROC curve and ROC area   
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= num_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
                 
    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    
    if method:
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(num_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    #plt.savefig("../images/ROC/ROC_5分类.png")
    plt.savefig(fold_pic + '/ROC_' + str(num_classes) + '.png')
    plt.show()
    
    
    
def loss_acc(history, save_folder_final):
    
    # plot the accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Mode Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(save_folder_final + '/Mode_accuracy.png')                
    plt.show()
                    
    # plot the loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Mode loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(save_folder_final + '/Mode_loss.png')
    plt.show()
                        
    

def save_csv_performance(testLabel, predict_label_test, predict_label_prob, save_folder_final):
    
    temp_cm = confusion_matrix(testLabel, predict_label_test)
    
    accuracy_result = accuracy_score(testLabel, predict_label_test)
    fscore_result = f1_score(testLabel, predict_label_test, average='weighted')    
    f1_score_class = f1_score(testLabel, predict_label_test, average=None)
    kapppa_score = cohen_kappa_score(testLabel, predict_label_test)
            
    np.savetxt(save_folder_final + '/accuracy_fscore_kappa.csv', [accuracy_result, fscore_result, kapppa_score], delimiter=',');
    np.savetxt(save_folder_final + '/f1_score_class.csv', f1_score_class, delimiter=',');
    np.savetxt(save_folder_final + '/confusion_matrix.csv', temp_cm, delimiter=',');
    np.savetxt(save_folder_final + '/predict_label.csv', predict_label_test, delimiter=',')
    np.savetxt(save_folder_final + '/testLabel.csv', testLabel, delimiter=',')
    np.savetxt(save_folder_final + '/predict_label_prob.csv', predict_label_prob, delimiter=',')


    
    
    
    
    
    
    
    
    