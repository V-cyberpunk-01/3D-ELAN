import dgl
# from dgl.nn.pytorch.conv import
# from torch_geometric.nn import GCNConv,GATConv,SAGEConv
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
from torch_geometric.nn import GCNConv,GATConv,SAGEConv
import numpy as np
import math
from torch.nn.parameter import Parameter
import alignn
from GAT import GraphAttentionLayer,SpecialSpmmFunction,SpecialSpmm,SpGraphAttentionLayer
from GCN import GCN

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class GraphSAGE(torch.nn.Module):
    def __init__(self, feature, hidden1,hidden2, out,node_num):
        super(GraphSAGE, self).__init__()
        self.sage1 = SAGEConv(feature, hidden1)
        self.sage2 = SAGEConv(hidden1, hidden2)
        self.sage3 = SAGEConv(hidden2, out)
        self.weight1 = nn.Parameter(torch.FloatTensor(1,node_num)).to(device)
        self.weight2 = nn.Parameter(torch.FloatTensor(out, 64)).to(device)
        nn.init.xavier_uniform_(self.weight1)
        nn.init.xavier_uniform_(self.weight2)

    def forward(self, features, edges):
        features = self.sage1(features, edges)
        features = F.gelu(features)
        features = F.dropout(features, training=self.training)

        features = self.sage2(features, edges)
        features = F.gelu(features)
        features = F.dropout(features, training=self.training)

        features = self.sage3(features, edges)
        features = F.log_softmax(features, dim=1).squeeze(0)

        readout = torch.mm(self.weight1 , features)
        readout = torch.mm(readout , self.weight2)

        return readout

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
            x = q
            y = self.self_attention_norm(x)
            y = self.self_attention(y, y, y, attn_bias)
            y = self.self_attention_dropout(y)
            x = x + y

            y = self.ffn_norm(x)
            y = self.ffn(y)
            y = self.ffn_dropout(y)
            
            x = x + y

        else:
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

class TriELAN(nn.Module):
    def __init__(self, hidden_size,
                 ffn_size,
                 out_size,
                 dropout_rate,
                 readout_attn,
                 attention_dropout_rate,
                 num_heads,
                 wo_extra_feature=False,
                 attention_layers: int =3,
                 com_size=64,
                 node_num=10,
                 tri_shape=[16,20],
                 dropout=0.5):
        super(TriELAN, self).__init__()

        myconf = myconfig()

        self.ali = alignn.ALIGNN(myconf)
        adjusted_hidden_size = 4 * hidden_size if not wo_extra_feature else 2 * hidden_size

        self.wo_extra_feature = wo_extra_feature
        self.readout_attn = readout_attn

        self.encoder1 = GCN(nfeat = 85,nhid = 512,node_num = node_num, nclass = 256, nout=hidden_size) # all = 23
        self.encoder2 = GCN(nfeat = 85,nhid = 512,node_num = node_num, nclass = 256, nout=hidden_size)

        self.ffn_norm1 = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, num_heads)
        self.ffn1 = FeedForwardNetwork(hidden_size,ffn_size, dropout_rate)
        self.ffn_dropout1 = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm( 2 * hidden_size )
        self.ffn = FeedForwardNetwork(2* hidden_size, 2*ffn_size ,dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

        self.linear_com = nn.Linear(com_size, hidden_size)
        self.com_relu = nn.SiLU()

        self.self_attention_norm_com = nn.LayerNorm(com_size)
        self.self_attention_com = MultiHeadAttention(com_size, attention_dropout_rate, num_heads=8)
        self.ffn_com = FeedForwardNetwork(com_size,com_size ,dropout_rate)

        self.ffn_com_dropout = nn.Dropout(dropout_rate)

        # triplet
        self.self_attention_norm_tri = nn.LayerNorm(hidden_size)
        self.self_attention_tri = MultiHeadAttention(hidden_size, attention_dropout_rate, num_heads=8)
        self.ffn_tri = FeedForwardNetwork(hidden_size, hidden_size, dropout_rate)
        self.ffn_tri_dropout = nn.Dropout(dropout_rate)

        # projection for com
        self.com_proj = MLPLayer(com_size, hidden_size)

        # for angle info using a MLP to do proj
        self.tri_proj = nn.Sequential(
            MLPLayer(tri_shape[0]*tri_shape[1], 2 *hidden_size),
            nn.Dropout(dropout_rate),
            MLPLayer(2 * hidden_size, 1 * hidden_size),
            nn.Dropout(dropout_rate),
        )

        # ---------
        # decoder
        # ---------
        self.res_all_norm = nn.LayerNorm( 4 * hidden_size )
        self.res_all_attention = MultiHeadAttention( 4 * hidden_size, attention_dropout_rate, num_heads)
        # self.res_all_ffn = FeedForwardNetwork( 4 * hidden_size, 4 * hidden_size ,dropout_rate)
        self.res_all_dropout = nn.Dropout(dropout_rate)

        self.res_all_norm1 = nn.LayerNorm(4 * hidden_size)
        self.res_all_attention1 = MultiHeadAttention(4 * hidden_size, attention_dropout_rate, num_heads)
        # self.res_all_ffn1 = FeedForwardNetwork( 4 * hidden_size, 4 * hidden_size ,dropout_rate)
        self.res_all_dropout1 = nn.Dropout(dropout_rate)

        self.res_all_norm12 = nn.LayerNorm(4 * hidden_size)
        self.res_all_attention12 = MultiHeadAttention(4 * hidden_size, attention_dropout_rate, num_heads)
        # self.res_all_ffn2 = FeedForwardNetwork( 4 * hidden_size, 4 * hidden_size ,dropout_rate)
        self.res_all_dropout12 = nn.Dropout(dropout_rate)
        
        self.attention_blocks = nn.ModuleList([
            self._build_attention_block(adjusted_hidden_size, attention_dropout_rate, num_heads)
            for _ in range(attention_layers)
        ])

        if wo_extra_feature:
            self.linear_proj = nn.Linear(2 * hidden_size, hidden_size * 2, dropout_rate)
        else:
            self.linear_proj = nn.Linear(4 * hidden_size, hidden_size * 2, dropout_rate)
        
        self.all_proj = nn.Linear(3 * hidden_size + com_size, 4 * hidden_size)
        
        self.linear_dropout = nn.Dropout(dropout_rate)
        self.res_all_norm2 = nn.LayerNorm(2 * hidden_size)
        self.res_all_attention2 = MultiHeadAttention(2 * hidden_size, attention_dropout_rate, num_heads)
        self.res_all_ffn2 = FeedForwardNetwork(2 * hidden_size, 2 * hidden_size, dropout_rate)
        self.res_all_dropout2 = nn.Dropout(dropout_rate)

        self.out_proj = nn.Sequential(
            nn.LayerNorm(2 * hidden_size),
            nn.Linear(2 * hidden_size, hidden_size, dropout_rate),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, out_size),
        )
        
        self.ln_2 = nn.LayerNorm(2 * hidden_size)
        self.ff = nn.Linear(2 * hidden_size, 2 * hidden_size)
        
        self.out_proj_old = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size, dropout_rate),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, out_size),
        )

        self.out_proj_dir = nn.Sequential(
            nn.Linear(4 * hidden_size, hidden_size, dropout_rate),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, out_size),
        )
        
    def _build_attention_block(self, hidden_size, dropout_rate, num_heads):
        """to build attentio block for information augmentation"""
        attention = MultiHeadAttention(hidden_size, dropout_rate, num_heads)
        norm = nn.LayerNorm(hidden_size)
        ffn = FeedForwardNetwork(hidden_size, hidden_size, dropout_rate)
        dropout = nn.Dropout(dropout_rate)
        return nn.ModuleDict({
            'attention': attention,
            'norm': norm,
            'ffn': ffn,
            'dropout': dropout
        })
        
    def initialize_weights(self):
        if self.activation is None:
            nn.init.xavier_uniform_(self.weight)
        else:
            nn.init.kaiming_uniform_(self.weight, nonlinearity='leaky_relu')
        if self.use_bias:
            nn.init.zeros_(self.bias)

    def get_triplet_readout(self,dimA,dimB):
        self.weight1 = nn.Parameter(torch.FloatTensor(32, dimA))
        self.weight2 = nn.Parameter(torch.FloatTensor(dimB, 32))


    def forward(self, batchg, batchS, batchT, attn_bias=None):
        '''
        :param batchg (non-alignn): Invariant graph with replaced nodes, containing three components:
                               graph node features (feature), adjacency matrix of the graph (adj),
                               and global graph features (com).
        :param batchg (alignn):
        '''

        graph_features = torch.FloatTensor([batchg[i][0] for i in range(len(batchg))]).to(device)
        graph_adjs = torch.FloatTensor([batchg[i][1] for i in range(len(batchg))]).to(device)
        graph_com = torch.FloatTensor([batchg[i][2] for i in range(len(batchg))]).to(device)

        shell_features = torch.FloatTensor([batchS[i][0] for i in range(len(batchg))]).to(device)
        shell_adjs = torch.FloatTensor([batchS[i][1] for i in range(len(batchg))]).to(device)
        shell_com = torch.FloatTensor([batchS[i][2] for i in range(len(batchg))]).to(device)

        # obtain bond and angle information
        triplet = (batchT[0].to(device), batchT[1].to(device))

        graph_embedding = self.encoder1(graph_features, graph_adjs)
        shell_embedding = self.encoder2(shell_features, shell_adjs)

        triplet_embedding = self.ali(triplet)

        # ---------------------------------------------------------
        # subgraph1 / subgraph2 / global_feature / triplet feature
        # ---------------------------------------------------------
        x1 = graph_embedding
        x2 = shell_embedding
        # x3 = self.com_proj(graph_com)
        x3 = graph_com
        x4 = self.tri_proj(triplet_embedding.view(-1).unsqueeze(0))

        # subgraph1 H
        x1 = self.ffn_norm1(x1)
        y1 = self.self_attention(x1, x1, x1, attn_bias)
        y1 = self.ffn_dropout1(y1)
        x1 = x1 + y1

        # subgraph2 H
        x2 = self.ffn_norm1(x2)
        y2 = self.self_attention(x2, x2, x2, attn_bias)
        y2 = self.ffn_dropout1(y2)
        x2 = x2 + y2

        # statement feature
        x3 = self.self_attention_norm_com(x3)
        y3 = self.self_attention_com(x3,x3,x3,attn_bias)
        y3 = self.ffn_com_dropout(y3)
        x3 = x3 + y3

        # Obtaining the information matrix of bond and angle triplets with dim of 1 H
        x4 = self.self_attention_norm_tri(x4)
        y4 = self.self_attention_tri(x4, x4, x4, attn_bias)
        # y4 = self.ffn_tri(y4)
        y4 = self.ffn_tri_dropout(y4)
        x4 = x4 + y4

        # concat the representation
        if self.wo_extra_feature:
            x = torch.cat((x1,x2),dim=2)
        else:
            x = torch.cat((x1,x2,x3,x4),dim=2)
            if x1.shape!=x3.shape:
                x = self.all_proj(x)
        

        # ------------
        # readout
        # ------------
        
        if self.readout_attn == True:
            for block in self.attention_blocks:
                x = block['norm'](x)
                y = block['attention'](x, x, x, attn_bias)
                y = block['dropout'](y)
                x = x + y
            
            # if with extro feature, using this to projection the dim 
            if self.wo_extra_feature == False:
                x = self.linear_proj(x)
                # y = self.ff(self.ln_2(y)) + y
                x = self.linear_dropout(x)

                # add a attn block for the projection
                y = self.res_all_norm2(x)
                y = self.res_all_attention2(x, x, x, attn_bias)  
                y = self.res_all_ffn2(y)  
                y = self.res_all_dropout2(y)
                x = x + y

        # output projection
        if self.readout_attn == True:
            x = self.out_proj_old(x)
        else:
            if self.wo_extra_feature == True:
                x = self.out_proj_old(x)
            elif self.wo_extra_feature == False:
                x = self.out_proj_dir(x)

        return x