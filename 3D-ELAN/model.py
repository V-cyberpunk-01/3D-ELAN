import dgl
from dgl.data import MiniGCDataset
import networkx as nx
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
import numpy as np
import math
from torch.nn.parameter import Parameter
import alignn
from torch_geometric.nn import GCNConv,GATConv,SAGEConv
from torch_geometric.datasets import Planetoid
from GAT import GraphAttentionLayer,SpecialSpmmFunction,SpecialSpmm,SpGraphAttentionLayer

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, node_num, nout, dropout=0.5):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.out = nout
        self.dropout = dropout
        self.predict = nn.Linear(nclass * node_num, nout)
        # self.predict = nn.Linear(nclass, nout)

    def forward(self, xs, adjs):
        batch_size = xs.shape[0]
        embedding = torch.tensor(np.zeros(shape=(batch_size, self.out), dtype=np.float32)).to(device)

        for i in range(batch_size):
            x = xs[i]
            adj = adjs[i]

            x = F.relu(self.gc1(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gc2(x, adj)

            # 原本这里算出一个值
            # x = torch.mean(self.predict(x.unsqueese()))
            x = self.predict(x.view(-1))

            # 把最后得到的矩阵，第一列取出来变成x1，后面的做mean，变成x2
            # center = x[0]
            # other = x[1:].mean(0)
            # x = center + other
            # x = self.predict(x)

            embedding[i] = x

        return embedding

# -------------
# GAT embedding
# -------------
class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        ## 定义两层的注意力卷积
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_dim = nclass

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        self.predict = nn.Linear(nclass, nclass, bias=True)


    def forward(self, xs, adjs):
        '''
        :param x: 节点特征 [batch size,node num, feature dim]
        :param adj: 图结构 [batch size, node num, feature dim]
        :return:
        '''
        batch_size = xs.shape[0]
        embedding = torch.tensor(np.zeros(shape=(batch_size, self.out_dim), dtype=np.float32)).to(device)
        for i in range(batch_size):
            x = xs[i]
            adj = adjs[i]
            x = F.dropout(x, self.dropout, training=self.training)
            x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
            x = F.dropout(x, self.dropout, training=self.training)
            x = F.elu(self.out_att(x, adj))

            x = torch.mean(self.predict(x).squeeze(),dim=0)

            embedding[i] = x

        return embedding

class GraphSAGE(torch.nn.Module):
    def __init__(self, feature, hidden, classes):
        super(GraphSAGE, self).__init__()
        self.sage1 = SAGEConv(feature, hidden)
        self.sage2 = SAGEConv(hidden, classes)

    def forward(self, features, edges):
        features = self.sage1(features, edges)
        features = F.relu(features)
        features = F.dropout(features, training=self.training)
        features = self.sage2(features, edges)
        return F.log_softmax(features, dim=1)

# ------------
# CSGraphormer
# ------------

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        # assert x.size() == orig_q_size
        return x

class MLPLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        """Linear, Batchnorm, SiLU layer."""
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            # nn.BatchNorm1d(out_features),
            nn.SiLU(),)
    def forward(self, x):
        """Linear, Batchnorm, silu layer."""
        return self.layer(x)

class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, out_size,dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

        self.out_proj = nn.Linear(hidden_size, out_size)

    def forward(self, q, k, v, attn_bias=None):
        if(q.equal(k)):
            #残差连接计算只有一个输入的时候
            x = q
            y = self.self_attention_norm(x)
            y = self.self_attention(y, y, y, attn_bias)
            y = self.self_attention_dropout(y)
            # 残差
            x = x + y

            y = self.ffn_norm(x)
            y = self.ffn(y)
            y = self.ffn_dropout(y)
            # 残差
            x = x + y

        else:
            # 有三个输入的时候不用残差
            q = self.self_attention_norm(q)
            k = self.self_attention_norm(k)
            v = self.self_attention_norm(v)

            x = self.self_attention(q,k,v,attn_bias)
            y = self.ffn_norm(x)
            y = self.ffn(y)
            y = self.ffn_dropout(y)
            # 残差
            x = x + y

        x = self.out_proj(x)

        return x


def accuracy(output, y):
    return (output.argmax(1) == y).type(torch.float32).mean().item()



class myconfig():
    def __init__(self):
        self.classification = False
        self.atom_input_features = 85
        self.hidden_features = 128
        self.edge_input_features = 20
        self.embedding_features = 128
        self.gcn_layers = 2
        self.link = 'identity'
        self.triplet_input_features = 20
        self.alignn_layers = 2
        self.output_features = 64

class CSGraphormer(nn.Module):
    def __init__(self, hidden_size,
                 ffn_size,
                 out_size,
                 dropout_rate,
                 attention_dropout_rate,
                 num_heads,
                 com_size = 291,
                 tri_shape = [2730,20],

                 dropout=0.5):
        super(CSGraphormer, self).__init__()

        myconf = myconfig()

        self.ali = alignn.ALIGNN(myconf)

        self.encoder1 = GraphSAGE(85, 128, hidden_size)
        self.encoder2 = GraphSAGE(85, 128, hidden_size)
        # self.encoder1 = GCN(nfeat = 85,nhid = 300,node_num = 15, nclass = 128, nout=64) # all = 23
        # self.encoder2 = GCN(nfeat = 85,nhid = 300,node_num = 15, nclass = 128, nout=64)
        self.encoder3 = GAT(nfeat = 85,nhid = 256, nclass = 64, dropout=0.5, alpha=0.5, nheads=8)


        self.fuse = nn.Linear(hidden_size*2,hidden_size)

        # 孪生网络的注意力残差
        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        # 合并之后的注意力残差
        self.ffn_norm = nn.LayerNorm(2 * hidden_size )
        self.ffn = FeedForwardNetwork(2* hidden_size, 2*ffn_size ,dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

        # 对全局特征所作的残差连接
        self.self_attention_norm_com = nn.LayerNorm(com_size)
        self.self_attention_com = MultiHeadAttention(com_size, attention_dropout_rate, num_heads=8)
        self.linear_com = nn.Linear(com_size, hidden_size)
        self.linear_com_dropout = nn.Dropout(dropout_rate)

        # triplet
        self.self_attention_norm_tri = nn.LayerNorm(hidden_size)
        self.self_attention_tri = MultiHeadAttention(hidden_size, attention_dropout_rate, num_heads=8)
        self.linear_tri = nn.Linear(hidden_size, hidden_size)
        self.linear_tri_dropout = nn.Dropout(dropout_rate)

        # 对键角信息进行处理 2*hidden state --- 用矩阵乘
        self.A, self.B = nn.Parameter(torch.FloatTensor(512, tri_shape[0])), \
                         nn.Parameter(torch.FloatTensor(tri_shape[1], 64))
        self.C, self.D = nn.Parameter(torch.FloatTensor(256, 512)), \
                         nn.Parameter(torch.FloatTensor(64, 16))
        self.E, self.F = nn.Parameter(torch.FloatTensor(128, 256)), \
                         nn.Parameter(torch.FloatTensor(16, 1))
        nn.init.xavier_uniform_(self.A)
        nn.init.xavier_uniform_(self.B)
        nn.init.xavier_uniform_(self.C)
        nn.init.xavier_uniform_(self.D)
        nn.init.xavier_uniform_(self.E)
        nn.init.xavier_uniform_(self.F)

        # 对键角信息 用linear层
        self.tri_proj = nn.Sequential(
            MLPLayer(tri_shape[0]*tri_shape[1], 16*hidden_size),
            nn.Dropout(dropout_rate),
            MLPLayer(16*hidden_size, 8*hidden_size),
            nn.Dropout(dropout_rate),
            MLPLayer(8 * hidden_size, 1 * hidden_size),
            nn.Dropout(dropout_rate),
        )

        # 全局attention resnet 1
        self.res_all_norm = nn.LayerNorm(4*hidden_size)
        self.res_all_attention = MultiHeadAttention(4*hidden_size, attention_dropout_rate, num_heads)
        self.res_all_ffn = FeedForwardNetwork( 4 * hidden_size, 4 * hidden_size ,dropout_rate)
        self.res_all_dropout = nn.Dropout(dropout_rate)

        self.linear_proj = nn.Linear(hidden_size * 4, hidden_size * 2, dropout_rate)
        self.linear_dropout = nn.Dropout(dropout_rate)

        # 全局 resnet2
        self.res_all_norm2 = nn.LayerNorm(2 * hidden_size)
        self.res_all_attention2 = MultiHeadAttention(2 * hidden_size, attention_dropout_rate, num_heads)
        self.res_all_ffn2 = FeedForwardNetwork(2 * hidden_size, 2 * hidden_size, dropout_rate)
        self.res_all_dropout2 = nn.Dropout(dropout_rate)


        self.out_proj = nn.Sequential(
            nn.Linear(2*hidden_size , hidden_size, dropout_rate),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, out_size),
        )

    def initialize_weights(self):
        if self.activation is None:
            nn.init.xavier_uniform_(self.weight)
        else:
            nn.init.kaiming_uniform_(self.weight, nonlinearity='leaky_relu')
        if self.use_bias:
            nn.init.zeros_(self.bias)

    def forward(self, batchg,labelG, batchS,labelS,batchT, attn_bias=None):
        '''
        :param batchg(非alignn): 带有替换节点的等变图，里面包括了三个部分；图节点的特征（feature），图的邻接矩阵（adj），图的全图特征（com）
        :param batchg(alignn):
        '''

        # 用alignn

        # batchS, label = (batchS[0].to(device), batchS[1].to(device)), labelS.to(device)
        # label = torch.FloatTensor(labelS).squeeze(0).to(device)
        #
        # triplet_embedding = self.ali(batchg).unsqueeze(0)

        # 不用alignn的时候
        graph_features = torch.FloatTensor([batchg[i][0] for i in range(len(batchg))]).to(device)
        graph_adjs = torch.FloatTensor([batchg[i][1] for i in range(len(batchg))]).to(device)
        graph_com = torch.FloatTensor([batchg[i][2] for i in range(len(batchg))]).to(device)

        shell_features = torch.FloatTensor([batchS[i][0] for i in range(len(batchg))]).to(device)
        shell_adjs = torch.FloatTensor([batchS[i][1] for i in range(len(batchg))]).to(device)
        shell_com = torch.FloatTensor([batchS[i][2] for i in range(len(batchg))]).to(device)

        # alignn 完整图 获得键角信息
        triplet, _ = (batchT[0].to(device), batchT[1].to(device)), labelG.to(device)

        graph_embedding = self.encoder1(graph_features, graph_adjs)
        shell_embedding = self.encoder2(shell_features, shell_adjs)
        triplet_embedding = self.ali(triplet)


        # x = self.fuse(torch.cat((shell_embedding,graph_embedding),1))
        # TODO: 加入线性变换，分开用linear乘了之后拼在一起
        # x = torch.cat((shell_embedding,graph_embedding),1)

        # -------------------------------------
        # 子图1 子图2  成分信息  键角信息 embedding
        # -------------------------------------
        x1 = graph_embedding
        x2 = shell_embedding
        x3 = graph_com
        x4 = self.tri_proj(triplet_embedding.view(-1).unsqueeze(0))

        # 子图1 H
        y1 = self.self_attention_norm(x1)
        y1 = self.self_attention(y1, y1, y1, attn_bias)
        y1 = self.self_attention_dropout(y1)
        # 残差
        x1 = x1 + y1

        # 子图2 H
        y2 = self.self_attention_norm(shell_embedding)
        y2 = self.self_attention(y2, y2, y2, attn_bias)
        y2 = self.self_attention_dropout(y2)
        x2 = x2 + y2

        # 全局特征 attention resnet 291 -- 291
        x3 = self.self_attention_norm_com(x3)
        y3 = self.self_attention_com(x3,x3,x3,attn_bias)
        y3 = self.self_attention_dropout(y3)
        x3 = x3 + y3

        x3 = self.linear_com(x3) # H
        # x3 = y3.unsqueeze(0)

        # 键角 三元组 的信息矩阵 1 H
        y4 = self.self_attention_norm_tri(x4)
        y4 = self.self_attention(y4, y4, y4, attn_bias)
        y4 = self.self_attention_dropout(y4)
        x4 = x4 + y4


        # 聚合最后的结果

        x = torch.cat((x1,x2,x3,x4),dim=2)
        # x = torch.cat((x1, x2), dim=2)
        # x = x1 + x2

        # ------------
        # readout
        # ------------
        # 残差1 attention 64*3 -- 64*3  0.701
        # 残差1 ffn 64 * 3 -- 64 * 3 0.732 0.709
        y = self.res_all_norm(x)
        y = self.res_all_attention(y,y,y,attn_bias) # attention 残差
        y = self.res_all_ffn(y)   # ffn 残差
        y = self.res_all_dropout(y)
        x = x + y

        # 前向 ffn 64*3 - 64*2
        y = self.linear_proj(x)
        x = self.linear_dropout(y)

        # 残差3 attention 64 -- 64
        y = self.res_all_norm2(x)
        y = self.res_all_attention2(y, y, y, attn_bias)  # attention 残差
        y = self.res_all_ffn2(y)  # ffn 残差
        y = self.res_all_dropout2(y)
        x = x + y

        # 前向


        # 残差

        # 输出 64 -- 32 -- 1
        x = self.out_proj(x)

        return x