# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 17:47:21 2021

@author: arnou
"""

def wavelet_node(level):
                           
    node = []
    
    if level == 1:
        node = ['a', 'd']

    elif level == 2:        
        node1 = ['a', 'd']
        for i1 in range(2):                                           
            node2 = ['a', 'd']                        
            for i2 in range(2):                            
                node.append(node1[i1] + node2[i2])
                    
    elif level == 3:
        node1 = ['a', 'd']
        for i1 in range(2):                                           
            node2 = ['a', 'd']                        
            for i2 in range(2):                            
                node3 = ['a', 'd']
                for i3 in range(2):
                    node.append(node1[i1] + node2[i2] + node3[i3])  
                    
    elif level == 4:
        node1 = ['a', 'd']
        for i1 in range(2):                                           
            node2 = ['a', 'd']                        
            for i2 in range(2):                            
                node3 = ['a', 'd']
                for i3 in range(2):
                    node4 = ['a', 'd']
                    for i4 in range(2): 
                        node.append(node1[i1] + node2[i2] + node3[i3] + node4[i4])
                                                                        
    elif level == 5:
        node1 = ['a', 'd']
        for i1 in range(2):                                           
            node2 = ['a', 'd']                        
            for i2 in range(2):                            
                node3 = ['a', 'd']
                for i3 in range(2):
                    node4 = ['a', 'd']
                    for i4 in range(2): 
                        node5 = ['a', 'd']
                        for i5 in range(2):
                            node.append(node1[i1] + node2[i2] + node3[i3] + node4[i4] + node5[i5])   
                                    
    elif level == 6:
        node1 = ['a', 'd']
        for i1 in range(2):                                           
            node2 = ['a', 'd']                        
            for i2 in range(2):                            
                node3 = ['a', 'd']
                for i3 in range(2):
                    node4 = ['a', 'd']
                    for i4 in range(2): 
                        node5 = ['a', 'd']
                        for i5 in range(2):
                            node6 = ['a', 'd']
                            for i6 in range(2):
                                node.append(node1[i1] + node2[i2] + node3[i3] + node4[i4] + node5[i5] + node6[i6]) 
    else:
        print('Please Input value 1-6 for WPD level')
                                                              
    return node






