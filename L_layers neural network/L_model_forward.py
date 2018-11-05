#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 20:40:20 2018

@author: jiangsiyang
"""

from Linear_activation_forward import linear_activation_forward
import numpy as np
def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A 
        ### START CODE HERE ### (≈ 2 lines of code)
        A, cache = linear_activation_forward(A_prev, parameters["W"+str(l)], parameters["b"+str(l)], activation = "relu")
        caches.append(cache)
        ### END CODE HERE ###
    
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    # Notice that the last activation function is sigmoid
    
    AL, cache = linear_activation_forward(A, parameters["W"+str(L)], parameters["b"+str(L)], activation = "sigmoid")
    # 这里有一点点绕， 需要看明白整个过程才能看懂
    caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))
    return AL, caches

def predict(params, X, y):
    
    preds = np.zeros((1,X.shape[1]))
    probs, caches = L_model_forward(X,params)

    for i in range(X.shape[1]):
        if probs[0, i] >= 0.5:
            preds[0, i] = 1

    preds = np.squeeze(preds)
    acc = np.mean(preds == y)
    
    

    return acc


