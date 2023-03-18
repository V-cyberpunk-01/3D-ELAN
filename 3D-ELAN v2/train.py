import os
import math
import pandas as pd
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import alignn

from model_ffnresnet import GCN,EncoderLayer,TriELAN
from process import data_process
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from utils import draw_epoches,draw_regression_curve,accuracy,evaluate

data_path = './1018/NbSi_c_c43_f50_k10/all_connect'

adj_all_path = str(data_path + '/adj/')
feature_all_path = str(data_path + '/feature/')
com_feature_path_g1 = './0916/composition/composition_feature_c_pca.csv'

adj_shell_path = str(data_path + '/adj/')
feature_shell_path = str(data_path + '/feature/')
com_feature_path_g2 = './0916/composition/composition_feature_c_pca.csv'


# --------
# 拿到处理好的数据并划分数据集
# --------
# 1.拿到数据集：输入的依次是图的adj、节点特征矩阵、全局成分特征
train_all_loader, valid_all_loader, test_all_loader = data_process(adj_all_path,
                                                                   feature_all_path,
                                                                   com_feature_path_g1,
                                                                   model_name='gnn',
                                                                   mask=1,
                                                                   seed=8)  # 置换原子中心ego
train_shell_loader, valid_shell_loader, test_shell_loader = data_process(adj_shell_path,
                                                                         feature_shell_path,
                                                                         com_feature_path_g2,
                                                                         model_name='gnn',
                                                                         mask=0,
                                                                         seed=8)  # 背景原子图
train_triplet_loader, valid_triplet_loader, test_triplet_loader = data_process(adj_all_path,
                                                                               feature_all_path,
                                                                               com_feature_path_g1,
                                                                               model_name='alignn',
                                                                               mask=1,
                                                                               seed=8)  # 计算三元组用的ego

device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')

# -----
# MODEL
# -----

# 模型训练

decoder = TriELAN(hidden_size = 64,
                    ffn_size = 64,
                    out_size = 1,
                    dropout_rate = 0.8,
                    attention_dropout_rate=0.5,
                    num_heads=8,
                  node_num=10,  # 输入的节点num
                  tri_shape=[16,20]  # 对应键角编码的长度 [(node_num-2)*2,20]
                  ).to(device)

loss_func = nn.MSELoss().to(device)
# 定义Adam优化器
optimizer3 = optim.Adam(decoder.parameters(), lr=0.00001)
# optimizer3 = optim.RMSprop(decoder.parameters(),lr=0.00005)

epoch_num = 600

# 打开模型的训练
decoder.train()

epoch_losses = []
val_losses = []

for epoch in range(epoch_num):
    epoch_loss = 0
    val_loss = 0
    for iter, (graph,shell,triplet) in enumerate(zip(train_all_loader,train_shell_loader,train_triplet_loader)):

        batchg, labelG = graph[0],graph[1]
        batchS, labelS = shell[0],shell[1]
        batchT = triplet[0]

        label = th.FloatTensor(labelS).squeeze(0).to(device)
        optimizer3.zero_grad()

        # 集成起来训练
        prediction = decoder(batchg,batchS,batchT)

        loss = loss_func(prediction,label)

        loss.backward()
        optimizer3.step()
        epoch_loss += loss.detach().item()

    for iter, (all,shell,triplet) in enumerate(zip(valid_all_loader,valid_shell_loader,valid_triplet_loader)):

        batchg, labelG = all[0], all[1]
        batchS, labelS = shell[0], shell[1]
        batchT = triplet[0]

        label = th.FloatTensor(labelS).squeeze(0).to(device)

        decoder.eval()
        prediction = decoder(batchg, batchS, batchT)
        loss_val = F.mse_loss(prediction, label).item()

        val_loss += loss_val

    epoch_loss /= (len(train_all_loader) + 1)
    val_loss /= (len(valid_all_loader)+1)
    # if epoch % 10 == 0:
    print('Epoch {}, train_loss:{:.4f} val_loss:{:.4f}'.format(epoch, epoch_loss,val_loss))
    epoch_losses.append(epoch_loss)
    val_losses.append(val_loss)

draw_epoches(epoch_losses,'train_loss')
draw_epoches(epoch_losses,'val_loss')


# 验证模型
print("---------Entering testing set-------------")

decoder.eval()

test_preds, test_labels = [], []
with th.no_grad():
    for iter, (all,shell,triplet) in enumerate(zip(test_all_loader,test_shell_loader,test_triplet_loader)):

        batchg, labelG = all[0], all[1]
        batchS, labelS = shell[0], shell[1]
        batchT = triplet[0]

        label = th.FloatTensor(labelS).squeeze(0).to(device)
        prediction = decoder(batchg, batchS, batchT)

        pred_ = prediction.cpu()
        label = label.cpu()

        test_preds.append(pred_.item())
        test_labels.append(label.item())

# test_preds = [b for i in test_preds for b in i]
# test_labels = [a for j in test_labels for a in j]

r2 = r2_score(test_labels,test_preds)
mae = mean_absolute_error(test_labels,test_preds)
rmse = mean_squared_error(test_labels,test_preds,squared=False)
print('r2_score:',r2,'rmse_score:',rmse,'mae_score:',mae)


th.save(decoder.state_dict(),'./model_feature50/model_d_feature50_'+'{:3f}_'.format(r2)+'{:2f}_'.format(mae)+'{:2f}_'.format(rmse)+".pth")

draw_regression_curve(test_labels,test_preds,[r2,rmse,mae])

