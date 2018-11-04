#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 09:15:15 2018

@author: jiangsiyang
"""

import numpy as np

def sigmoid(z):
    s = 1/(1+np.exp(-z))  #用科学计算包numpy    
    return s

def initialize_with_zeros(dim):
    w = np.zeros((dim,1))  # 明白np.zeros的用法 维度整体作为参数输入
    b = 0
    assert(w.shape == (dim, 1)) #两个作为查错的函数
    assert(isinstance(b, float) or isinstance(b, int)) 
    return w, b

def propagate(w, b, X, Y):
    m = X.shape[1]
    # FORWARD PROPAGATION (FROM X TO COST)
    A = sigmoid(np.dot(w.T,X)+b)                                    # compute activation
    cost = -1/m *(np.sum(Y*np.log(A)+(1-Y)*np.log(1-A)))  #python *直接表示的就是乘    multiply是点乘 np.dot矩阵乘法
    # compute cost
    #这个bug有点难找啊
    # BACKWARD PROPAGATION (TO FIND GRAD)
    
    dw = 1/m*np.dot(X,(A-Y).T) # 这里的公式是由于sigmoid函数的导数形式决定的
    db = 1/m*np.sum(A-Y)       #不同的active function的backward函数不一样
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost