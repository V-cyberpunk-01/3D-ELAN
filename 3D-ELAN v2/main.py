import os
import time
import math
import argparse
import numpy as np

import torch
import torch.nn.functional as F

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import alignn

from model_ffnresnet import GCN,EncoderLayer,TriELAN
from process import data_process
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from utils import draw_epoches,draw_regression_curve,accuracy,evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # path definition
    parser.add_argument('--data_path', type=str, default='./1018/NbSi_a_c0_f50_k15/all_connect',
                        help='directory to the (a)Nbsi data.')
    parser.add_argument('--com_feature_path_ego', type=str, default='./1018/composition/composition_feature_a_pca.csv',
                        help='the component feature of ego graph')
    parser.add_argument('--com_feature_path_background', default='./1018/composition/composition_feature_a_pca.csv',
                        help='the component feature of background graph')
    # parser.add_argument('--model_name', type=str, default='gnn',
    #                     help='the baseline model in the framework to be trained on.')

    parser.add_argument('--train', type=int, default=1,
                        help='validation or model training.')
    parser.add_argument('--train_epoch', type=int, default=600,
                        help='validation or model training.')
    parser.add_argument("--lr", type=float, default=0.00001,
                        help="learning rate.")
    parser.add_argument('--random_seed', type=int, default=8,
                        help='random seed for data shuffle')

    # parameter of 3D-ELAN
    parser.add_argument('--TriELAN_hidden_size', type=int, default=64,
                        help='model hidden size')
    parser.add_argument('--TriELAN_ffn_size', type=int, default=64,
                        help='model MLP hidden size')
    parser.add_argument('--TriELAN_dropout_rate', type=float, default=0.8,
                        help='model dropout rate')
    parser.add_argument('--TriELAN_attention_dropout_rate', type=float, default=0.5,
                        help='dropout rate specially for downstream attention computing')
    parser.add_argument('--TriELAN_num_heads', type=int, default=8,
                        help='num heads for multi-head attention computing')
    parser.add_argument('--TriELAN_node_num', type=int, default=15,
                        help='node num of input graph')
    parser.add_argument('--TriELAN_tri_shape', type=np.array, default=[26,20],
                        help='node num of input graph')
    parser.add_argument('--TriELAN_model_save_path', type=str, default='./model_feature50/model_a_feature50_',
                        help='save path of trained model')

    args = parser.parse_args()

    adj_all_path = str(args.data_path + '/adj/')
    feature_all_path = str(args.data_path + '/feature/')

    adj_shell_path = str(args.data_path + '/adj/')
    feature_shell_path = str(args.data_path + '/feature/')

    train_all_loader, valid_all_loader, test_all_loader = data_process(adj_all_path,
                                                                       feature_all_path,
                                                                       args.com_feature_path_ego,
                                                                       model_name='gnn',
                                                                       mask=1,
                                                                       seed=args.random_seed)  # 置换原子中心ego
    train_shell_loader, valid_shell_loader, test_shell_loader = data_process(adj_shell_path,
                                                                             feature_shell_path,
                                                                             args.com_feature_path_background,
                                                                             model_name='gnn',
                                                                             mask=0,
                                                                             seed=args.random_seed)  # 背景原子图
    train_triplet_loader, valid_triplet_loader, test_triplet_loader = data_process(adj_all_path,
                                                                                   feature_all_path,
                                                                                   args.com_feature_path_ego,
                                                                                   model_name='alignn',
                                                                                   mask=1,
                                                                                   seed=args.random_seed)  # 计算三元组用的ego
    loss_func = nn.MSELoss().to(device)

    # training
    if args.train:

        print("---------------------------------")
        print("---------| Now training |--------")
        print("---------------------------------")

        epoch_losses = []
        val_losses = []

        decoder = TriELAN(hidden_size=args.TriELAN_hidden_size,
                          ffn_size=args.TriELAN_ffn_size,
                          out_size=1,
                          dropout_rate=args.TriELAN_dropout_rate,
                          attention_dropout_rate=args.TriELAN_attention_dropout_rate,
                          num_heads=args.TriELAN_num_heads,
                          node_num=args.TriELAN_node_num,  # 输入的节点num
                          tri_shape=[(args.TriELAN_node_num-2)*2,20] # 对应键角编码的长度 [(node_num-2)*2,20]
                          ).to(device)


        optimizer3 = optim.Adam(decoder.parameters(), lr=args.lr)

        decoder.train()

        for epoch in range(args.train_epoch):

            epoch_loss = 0
            val_loss = 0
            for iter, (graph, shell, triplet) in enumerate(zip(train_all_loader, train_shell_loader, train_triplet_loader)):
                batchg, labelG = graph[0], graph[1]
                batchS, labelS = shell[0], shell[1]
                batchT = triplet[0]

                label = th.FloatTensor(labelS).squeeze(0).to(device)
                optimizer3.zero_grad()

                # 集成起来训练
                prediction = decoder(batchg, batchS, batchT)

                loss = loss_func(prediction, label)

                loss.backward()
                optimizer3.step()
                epoch_loss += loss.detach().item()

            for iter, (all, shell, triplet) in enumerate(zip(valid_all_loader, valid_shell_loader, valid_triplet_loader)):
                batchg, labelG = all[0], all[1]
                batchS, labelS = shell[0], shell[1]
                batchT = triplet[0]

                label = th.FloatTensor(labelS).squeeze(0).to(device)

                decoder.eval()
                prediction = decoder(batchg, batchS, batchT)
                loss_val = F.mse_loss(prediction, label).item()

                val_loss += loss_val

            epoch_loss /= (len(train_all_loader) + 1)
            val_loss /= (len(valid_all_loader) + 1)
            # if epoch % 10 == 0:
            print('Epoch {}, train_loss:{:.4f} val_loss:{:.4f}'.format(epoch, epoch_loss, val_loss))
            epoch_losses.append(epoch_loss)
            val_losses.append(val_loss)

        draw_epoches(epoch_losses, 'train_loss')
        draw_epoches(epoch_losses, 'val_loss')

        print("---------------------------------")
        print("---------| Now testing |--------")
        print("---------------------------------")

        decoder.eval()

        test_preds, test_labels = [], []
        with th.no_grad():
            for iter, (all, shell, triplet) in enumerate(zip(test_all_loader, test_shell_loader, test_triplet_loader)):
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

        r2 = r2_score(test_labels, test_preds)
        mae = mean_absolute_error(test_labels, test_preds)
        rmse = mean_squared_error(test_labels, test_preds, squared=False)
        print('r2_score:', r2, 'rmse_score:', rmse, 'mae_score:', mae)

        torch.cuda.empty_cache()

        th.save(decoder.state_dict(),
                args.TriELAN_model_save_path + '{:3f}_'.format(r2) + '{:2f}_'.format(mae) + '{:2f}_'.format(
                    rmse) + ".pth")

        decoder.initialize_weights()

        draw_regression_curve(test_labels, test_preds, [r2, rmse, mae])

