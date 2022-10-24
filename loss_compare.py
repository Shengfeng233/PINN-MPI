import numpy as np

from pinn_model import *
import pandas as pd
import matplotlib.pyplot as plt

size_of_process = 9
filename_loss = './data/1_1.9/{:02d}loss.csv'
filename_loss_0 = filename_loss.format(0)
loss_0 = pd.read_csv(filename_loss_0, header=None).values
rows = loss_0.shape[0]
x = np.linspace(1, rows, rows).reshape(-1, 1)
total_losses = np.empty((rows, 0),dtype=float)
data_losses = np.empty((rows, 0),dtype=float)
eqa_losses = np.empty((rows, 0),dtype=float)
pseudo_losses = np.empty((rows, 0),dtype=float)
for i in range(size_of_process):
    file_loss = pd.read_csv(filename_loss.format(i), header=None)
    loss = file_loss.values
    total_loss = loss[:, 0].reshape(-1,1)
    data_loss = loss[:, 1].reshape(-1,1)
    eqa_loss = loss[:, 2].reshape(-1,1)
    pseudo_loss = loss[:, 3].reshape(-1, 1)
    total_losses = np.append(total_losses, total_loss, axis=1)
    data_losses = np.append(data_losses, data_loss, axis=1)
    eqa_losses = np.append(eqa_losses, eqa_loss, axis=1)
    pseudo_losses = np.append(pseudo_losses, pseudo_loss, axis=1)

list_legend = []
fig_size = 6, 5
font = {'weight': 'normal', 'size': 14}
# 画各个子网络总loss对比
fig1, ax1 = plt.subplots(figsize=fig_size)
for i in range(size_of_process):
    ax1.semilogy(x, total_losses[:, i], marker='o')
    list_legend.append('sub_net: %d' % i)

ax1.legend(list_legend, prop=font, loc='upper right')
ax1.set_title('Total_loss_sub_nets', fontdict={'weight':'normal','size': 15})
ax1.set_xlabel('EPOCH', fontdict={'weight':'normal','size': 15})
ax1.set_ylabel('Loss', fontdict={'weight':'normal','size': 15})
ax1.set_xlim(0, rows)
ax1.set_ylim((1e-5, 1e-0))
plt.show()
