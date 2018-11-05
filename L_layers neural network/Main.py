#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 20:18:46 2018

@author: jiangsiyang
"""

# Writing an L_layer Neural Network without regularization
# Using the same dataset(Recongize a cat)

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
from L_layer_model import L_layer_model
from L_model_forward import predict

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# 随机抽出来一张图片看看
index = np.random.randint(1,100)
plt.imshow(train_set_x_orig[index])

m_train = train_set_x_orig.shape[0] #训练集有多少样本
m_test = test_set_x_orig.shape[0] #测试集有多少样本
num_px = train_set_x_orig.shape[1]  #图片有多少维度(像素点之类的)


## 按照指定的格式展开 变成一行(列)，从而变成特征来回归
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T # -1 表示后面均为一行排列
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

train_set_x = train_set_x_flatten/255. #像素点 元素均为1-255之间 所以除以255做一个归一化，方便计算
test_set_x = test_set_x_flatten/255.

layers_dims = [12288, 20, 7, 5, 1] #  4-layer model

parameters = L_layer_model(train_set_x, train_set_y, layers_dims,
                           learning_rate = 0.0035, num_iterations = 3400,
                           print_cost = True)

#pred_train = predict(train_x, train_y, parameters)

result = predict(parameters, train_set_x, train_set_y)
print(result)