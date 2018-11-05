#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 21:03:01 2018

@author: jiangsiyang
"""

from Linear_backward import linear_backward
import numpy as np

## 不好理解 本质上是 dZ = dA * g'(Z^[l])
def sigmoid_backward(dA,activation_cache):
    Z = activation_cache
    s = 1.0 / ((1 + np.exp(-Z)))
    dZ = dA * s * (1-s)
    return dZ
    
'''
    if np.exp(-Z) > np.log(np.finfo(type(Z)).max):
        s = 0
    else:
        s = 1.0 / ((1 + np.exp(-Z)))
''' 



def relu_backward(dA,activation_cache):
    Z = activation_cache
    dZ = np.array(dA, copy=True)    
    dZ[Z <= 0] = 0
    assert(dZ.shape == Z.shape)
    return dZ


def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        ### START CODE HERE ### (≈ 2 lines of code)
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ,linear_cache) # 这里为什么写这个呢？ 这里是有一点儿迷 这里是单层model的backward
        ### END CODE HERE ###
        
    elif activation == "sigmoid":
        ### START CODE HERE ### (≈ 2 lines of code)
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ,linear_cache)
        ### END CODE HERE ###
    
    return dA_prev, dW, db