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

# class GCN(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, node_num, nout, dropout=0.5):
#         super(GCN, self).__init__()

#         self.gc1 = GraphConvolution(nfeat, nhid)
#         self.gc2 = GraphConvolution(nhid, nout)
#         self.out = nout
#         self.dropout = dropout
#         self.predict = nn.Linear(nclass * node_num, nout)

#     def forward(self, xs, adjs):
#         batch_size = xs.shape[0]
#         embedding = torch.tensor(np.zeros(shape=(batch_size, self.out), dtype=np.float32)).to(device)

#         for i in range(batch_size):
#             x = xs[i]
#             adj = adjs[i]
#             x = self.gc1(x, adj)
#             x = F.dropout(x, self.dropout, training=self.training)
#             x = self.gc2(x, adj)
            
#             # x = torch.mean(x,dim=-2)
#             x = self.predict(x.view(-1))
#             embedding[i] = x

#         return embedding

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from torch_geometric.nn import GCNConv as GraphConvolution  # 假设使用PyTorch Geometric

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, node_num, nout, readout="linear", dropout=0.5):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.readout = readout
        # Adjust the output dim of gc2 according to the readout strategy
        if self.readout == "mean":
            self.gc2 = GraphConvolution(nhid, nout)
        elif self.readout == "linear":
            self.gc2 = GraphConvolution(nhid, nclass)
            self.predict = nn.Linear(nclass * node_num, nout)
        else:
            raise ValueError("Unsupported readout strategy")
        
        self.out = nout
        self.dropout = dropout

    def forward(self, xs, adjs):
        batch_size = xs.shape[0]
        embedding = torch.zeros(size=(batch_size, self.out), dtype=torch.float32).to(xs.device)

        for i in range(batch_size):
            x = xs[i]
            adj = adjs[i]
            x = F.relu(self.gc1(x, adj)) 
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gc2(x, adj)
            
            if self.readout == "mean":
                x = torch.mean(x, dim=-2)  
            elif self.readout == "linear":
                x = x.view(-1)  
                x = self.predict(x)

            embedding[i] = x

        return embedding
