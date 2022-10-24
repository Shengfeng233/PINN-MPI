"""
数据预处理器
将庞大的csv文件转换为mat文件
转化目标100*100
此文件将数据进行无量纲化
"""
import numpy as np
import pandas as pd
import scipy.io as sp
from read_data import *

filename_raw_data = './stack_Re100.mat'
filename_mat = './stack_Re100_bound.mat'
x, y, u, v, feature_mat = read_2D_data_surround_without_p(filename_raw_data)
x_mat = x.numpy()
y_mat = y.numpy()
u_mat = u.numpy()
v_mat = v.numpy()
stack = np.concatenate((x_mat, y_mat, u_mat, v_mat), axis=1)


# 存储mat文件
sp.savemat(filename_mat, {'stack': stack})
print("OK")
