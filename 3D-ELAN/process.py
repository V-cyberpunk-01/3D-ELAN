import dgl

import networkx as nx
import matplotlib.pyplot as plt
import os
import pandas as pd
from math import inf
import numpy as np
import torch as th
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import  Dataset,DataLoader,TensorDataset,random_split,Subset
from dgl.data import MiniGCDataset
from dgl.nn.pytorch import GraphConv
from sklearn.metrics import accuracy_score
from xgboost import plot_importance
import sklearn.preprocessing as skpre
import numpy as np
from tqdm import tqdm
from scipy.sparse import coo_matrix, csr_matrix, diags, eye
import copy

device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')

# -----------------------------
# 1.Get different type of graph
# ------------------------------
def get_triplet_graph(adj, features):
    '''
    用来做三元组的图结构
    '''
    fea = []
    adj = adj.loc[adj['isMasked'] == 1]  # 只取masked行
    feature_idx = list(set(adj['src'])
                       | set(adj['tgt']))  # 获取id不重复,这里取[2:5]是为了测试,实际请删掉
    for i in feature_idx:
        fea.append(features[i + 1].split()[2:])  # 第一行是置换能  获取特定id行的特征，



    id_map = dict().fromkeys(feature_idx)
    for i in feature_idx:
        id_map[i] = feature_idx.index(
            i
        )  #获取id映射map保证id顺序连续，这里有一个前提，feature_idx里面的id是从小到大，为了放心可能需要加一个sort

    g = dgl.graph((list(map(lambda x: id_map[x], adj['src'])),
                   list(map(lambda x: id_map[x], adj['tgt']))))
    g.edata['weight'] = th.FloatTensor(list(adj['weight']))
    g.ndata['atom_features'] = th.FloatTensor(np.array(fea, dtype='float'))
    g.edata['r'] = th.FloatTensor(np.array(adj[['x', 'y', 'z']]))  # 边的空间向量
    return g


def get_mask_graph(adj, features, mask):
    '''
    以mask标志分别取得只和置换原子有联系的和不包含置换原子的
    :param adj:
    :param features:
    :param mask:
    :return:
    '''
    if mask == 1:
        adj.loc[adj[(adj['isMasked'] == 1)].index,'weight'] = 0 # 做mask
    else:
        adj.loc[adj[(adj['isMasked'] == 0)].index, 'weight'] = 0

    g = dgl.graph((adj['src'], adj['tgt']))

    g.edata['weight'] = th.FloatTensor(adj['weight'])  # th.cat([th.tensor(adj['weight']),th.tensor(adj['weight'])])
    g.edata['r'] = th.FloatTensor(np.array(adj[['x', 'y', 'z']]))  # 边的空间向量

    temp = [features[i+1].split()[2:] for i in range(g.num_nodes())]
    g.ndata['atom_features'] = th.FloatTensor(np.array(temp, dtype='float'))

    # g = dgl.add_self_loop(g)  # 要用dgl里面的conv时
    return g


def get_graphsage_graph(adj,features,mask):
    if mask == 1:
        adj.loc[adj[(adj['isMasked'] == 1)].index,'weight'] = 0 # 做mask
    else:
        adj.loc[adj[(adj['isMasked'] == 0)].index, 'weight'] = 0

    edges = []
    for i, x in enumerate(zip(adj['src'], adj['tgt'])):
        edges.append([x[0], x[1]])
    temp = [features[i + 1].split()[2:] for i in range(len(features)-1)]
    features = th.FloatTensor(np.array(temp, dtype='float'))
    edges = th.tensor(edges, dtype=th.int64).T

    g = dgl.graph((adj['src'], adj['tgt']))

    g.edata['weight'] = th.FloatTensor(adj['weight'])  # th.cat([th.tensor(adj['weight']),th.tensor(adj['weight'])])
    g.edata['r'] = th.FloatTensor(np.array(adj[['x', 'y', 'z']]))  # 边的空间向量

    g.ndata['atom_features'] = features
    return features,edges,g


def get_one_graph2(adj,features):
    # 用字典可以直接解决节点重复问题，那么字典里一定只有23个节点！ 从下到上 4+1+4+5+4+1+4=23个节点
    d = dict(zip(list(adj['src']),list(adj['atom_id1'])))
    fea = {}
    # 这里获取Nb 和 Si的原子特征
    for i in range(1,3):
        fea[features[i].split()[0]]=features[i].split()[1:]
    # 这里获取替换原子的特征
    for i in range(3,5):
        fea[features[i].split()[0][:-2]]=features[i].split()[1:]
    g = dgl.graph((adj['src'],adj['tgt']))
    g.edata['weight'] = th.FloatTensor(adj['weight'])
    temp = []
    for k,v in d.items():
        temp.append(fea[v[:-2]]) # v[:-2] Nb00 -> Nb
    g.ndata['atom_features'] = th.FloatTensor(np.array(temp,dtype='float'))
    return g


def get_line_graph(graphs):
  '''
  获得图的原子线图表示。
  '''
  train_data = []
  for g in tqdm(graphs,desc='building line graphs'):
      lg = g.line_graph(shared=True,backtracking=False)
      lg.apply_edges(compute_bond_cosines)
      train_data.append(lg)
  return train_data


def compute_bond_cosines(edges):
    """Compute bond angle cosines from bond displacement vectors."""
    # line graph edge: (a, b), (b, c)
    # `a -> b -> c`
    # use law of cosines to compute angles cosines
    # negate src bond so displacements are like `a <- b -> c`
    # cos(theta) = ba \dot bc / (||ba|| ||bc||)
    r1 = -edges.src["r"]
    r2 = edges.dst["r"]
    bond_cosine = th.sum(r1 * r2, dim=1) / (
        th.norm(r1, dim=1) * th.norm(r2, dim=1)
    )
    bond_cosine = th.clamp(bond_cosine, -1, 1)
    return {"h": bond_cosine}


def get_weight_matrix(g):
    m = np.zeros((g.num_src_nodes(), g.num_dst_nodes()))

    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            try:
                m[i, j] = - g.edges[i, j].data['weight']
            except Exception as e:
                # print(e)
                pass
    return m


def draw_graph(g):
    plt.figure(figsize=(14, 6))
    nxg = g.to_networkx()
    pos = nx.spring_layout(nxg)
    # nx.draw(, with_labels=True)
    color = ['r' for _ in range(8)] + ['y' for _ in range(6)] + ['g' for i in range(12)]
    edgewidth = list(map(lambda x: 1 / x, g.edges[:].data['weight']))

    nx.draw_networkx_edges(nxg, pos, width=edgewidth,
                           edge_color=color)  # 奇怪的是：edgewidth应该是一个列表，为什么可以直接拿来用呢？Python怎么判断哪个边用的是哪个权重值？
    nx.draw_networkx_nodes(nxg, pos)
    nx.draw_networkx_labels(nxg, pos)

    plt.show()


class normalizer():
    def __init__(self, x):
        self.matrix = np.array(x, dtype=float)
        # print(self.matrix.shape)

    def minmax_normalize(self, x=None):  # 定义函数，对x进行归一化
        scaler = skpre.MinMaxScaler()
        if x:
            return scaler.fit_transform(x, x)
        else:
            return scaler.fit_transform(self.matrix, self.matrix)

    def absmax_normalize(self, x=None):
        scaler = skpre.MaxAbsScaler()
        if x:
            return scaler.fit_transform(x, x)
        else:
            return scaler.fit_transform(self.matrix, self.matrix)

    def minmax_minus1to1_normalize(self, x=None):
        if x:
            mean = np.array([0.5] * x.shape[1], dtype=float).reshape(1, -1)
            return np.true_divide(self.minmax_normalize(x) - mean, mean)
        else:
            mean = np.array([0.5] * self.matrix.shape[1], dtype=float).reshape(1, -1)
            return np.true_divide(self.minmax_normalize() - mean, mean)

    def standard_normalize(self, x=None):
        scaler = skpre.StandardScaler()
        if x:
            return scaler.fit_transform(x, x)
        else:
            return scaler.fit_transform(self.matrix, self.matrix)

    def reverse_minmax_minus1to1_normalize(self, src_x=None, x: np.array = None):
        assert src_x is not None, 'source matrix should not be None'
        min = np.min(src_x, 0)
        max = np.max(src_x, 0)
        if x is not None:
            return x * (max - min) + min
        else:
            return self.matrix * (max - min) + min

    @staticmethod
    def lambda_max(arr, axis=None, key=None, keepdims=False):
        if callable(key):
            idxs = np.argmax(key(arr), axis)
            if axis is not None:
                idxs = np.expand_dims(idxs, axis)
                result = np.take_along_axis(arr, idxs, axis)
                if not keepdims:
                    result = np.squeeze(result, axis=axis)
                return result
            else:
                return arr.flatten()[idxs]
        else:
            return np.amax(arr, axis)

    def inverse_absmax_normalize(self, src_x=None, x: np.array = None):
        # print(x*self.lambda_max(src_x,0,key=np.abs))
        if x is not None:
            return x * np.max(np.abs(src_x), 0)
        else:
            return self.matrix * (np.max(np.abs(src_x), 0))

# ------------------------------------
# 2.重写dataset函数，定义不同的collate方法
# ------------------------------------
class My_dataset(Dataset):
    def __init__(self, graphs, labels):
        super().__init__()
        self.src = graphs
        self.trg = labels

    def __getitem__(self, index):
        # print(index)
        return self.src[index], self.trg[index]

    def __len__(self):
        return len(self.src)

def collate1(samples):
    '''
    参考collate1函数，区别在于本函数接受的是dgl.graph数据
    '''
    graphs, labels = map(list, zip(*samples))
    gs,lgs=list(map(list,zip(*graphs)))
    # dgl.batch 将一批图看作是具有许多互不连接的组件构成的大型图
    return (dgl.batch(gs),dgl.batch(lgs)),th.FloatTensor(labels)

def collate2(samples):
    # 输入参数samples是一个列表
    # 列表里的每个元素是图和标签对，如[(graph1, label1), (graph2, label2), ...]
    # 每个图包含三个部分：图的特征 （15，85），图的邻接矩阵表示（15，15），图的全局特征（291）
    # zip(*samples)是解压操作，解压为[(graph1, graph2, ...), (label1, label2, ...)]
    graphs, labels = map(list, zip(*samples))

    # dgl.batch 将一批图看作是具有许多互不连接的组件构成的大型图
    # return dgl.batch(graphs), th.tensor(labels, dtype=th.float32)
    return graphs, th.FloatTensor(labels)

def collate3(samples):
    '''
    获得Nb预测数据，调用dgl包装图模型
    '''
    # graphs = map(list, zip(*samples))
    gs,lgs = list(map(list,zip(*samples)))
    # dgl.batch 将一批图看作是具有许多互不连接的组件构成的大型图
    return (dgl.batch(gs),dgl.batch(lgs))

def collate4(samples):
    return samples


def get_train_val_test_data(datasets, labels,co_fn,split=[0.7,0.2,0.1],seed=33):
    '''
    分割数据集
    '''
    co_fn = globals()[co_fn]
    assert datasets is not None and len(datasets) > 0, '数据集不能为空'
    lens = len(datasets)

    assert abs(sum(split) - 1) < 1e-5 or sum(split) == lens, 'split分割总和不为1或len(datasets)'
    # if sum(split) != lens:
    split = list(map(lambda x: int(x * lens), split))
    split[-1] = lens - sum(split[:-1])  # 确保总和为所有数据量

    n_train, n_val, n_test = split
    np.random.seed(seed)
    idxs = np.random.permutation(lens)  # 将原有索引打乱顺序

    # 这里下次需要改以下
    # idxs = np.arange(0,lens)


    # 计算每个数据集的索引
    idx_train = th.LongTensor(idxs[:n_train])
    idx_val = th.LongTensor(idxs[n_train:n_train + n_val])
    idx_test = th.LongTensor(idxs[n_train + n_val: ])

    train_data, valid_data, test_data = [My_dataset([datasets[i] for i in j], [labels[i] for i in j]) for j in
                                         [idx_train, idx_val, idx_test]]

    # 用subset来截取固定范围的test集
    # train_datasets = My_dataset(datasets, labels)
    # all_len = len(train_datasets)
    # train_idx = int(all_len * 0.7)
    # val_idx = int(all_len * 0.9)
    # train_data, valid_data, test_data = Subset(train_datasets,range(0,train_idx)),Subset(train_datasets,range(train_idx,val_idx)),Subset(train_datasets,range(val_idx,len(train_datasets)))

    train_data_loader = DataLoader(train_data, batch_size=1, shuffle=True, collate_fn=co_fn)
    valid_data_loader = DataLoader(valid_data, batch_size=1, shuffle=True, collate_fn=co_fn)
    test_data_loader = DataLoader(test_data, batch_size=1, shuffle=True, collate_fn=co_fn)
    print('train:{},valid:{},test:{}'.format(len(train_data_loader), len(valid_data_loader), len(test_data_loader)))
    return train_data_loader, valid_data_loader, test_data_loader


def get_pre_data(datasets, labels,co_fn):
    co_fn = globals()[co_fn]
    assert datasets is not None and len(datasets) > 0, '数据集不能为空'
    lens = len(datasets)

    # lens = 10
    idxs = np.arange(0, lens)

    train_data = My_dataset([datasets[i] for i in idxs], [labels[i] for i in idxs])

    train_data_loader = DataLoader(train_data, batch_size=1, shuffle=False, collate_fn=co_fn)
    print('train:{}'.format(len(train_data_loader)))
    return train_data_loader


# -------------
# 数据预处理
# -------------
def data_process(adj_path, feature_path, com_feature_path,model_name,mask):

    lgraphs = []  # 用来保存计算键角信息的图
    graphs_mask = [] # 保存
    labels = []
    all_adj_m = []
    all_feature_m = []
    all_com_feature = []
    weight_m = None

    com_feature = pd.read_csv(com_feature_path)   # 成分信息要重新写或者读

    for i, f in enumerate(os.listdir(feature_path)):
      if i % 50 == 0:
        print('{}/{}:{}'.format(i,len(os.listdir(feature_path)),f))
      with open(feature_path+f,'r') as res:

        features = res.read().split('\n')
        adj = pd.read_csv(adj_path+'adj_'+f.split('.')[0].split('_')[-1]+'.csv')

        if model_name == 'graphsage':
            sage_features,sage_edge,g_mask = get_graphsage_graph(adj,features,mask)  # 使用graphsage，需要构造node idx的列表

        else:
            g_mask = get_mask_graph(adj,features,mask)  # gcn gat

        tri_g = get_triplet_graph(adj,features)  # 用来计算三元组的图结构
        graphs_mask.append(g_mask)
        lgraphs.append(tri_g)

        if weight_m is None:
          weight_m = get_weight_matrix(g_mask)

        if model_name == "graphsage":
            all_adj_m.append(sage_edge)  # 放入邻接表
        else:
            all_adj_m.append(weight_m)  # 放入邻接矩阵

        all_com_feature.append(com_feature.iloc[i].values) # 成分信息
        features_na = normalizer(g_mask.ndata['atom_features'].numpy())  # 将图上节点特征做norm
        all_feature_m.append(features_na.minmax_normalize())  # 节点特征矩阵归一化处理
        labels.append(float(features[0]))

    draw_graph(g_mask)

    if model_name == 'alignn':
        lgs = get_line_graph(lgraphs)  # 计算线图
        a = list(zip(lgraphs, lgs))
        train_data_loader, valid_data_loader, test_data_loader = get_train_val_test_data(a, labels, co_fn='collate1')
        # train_data_loader = get_pre_data(a, labels, co_fn='collate1')

    elif model_name == 'graphsage':
        a = list(zip(all_feature_m, all_adj_m, all_com_feature))
        labels = normalizer(np.array(labels).reshape(-1, 1)).absmax_normalize() * 10
        train_data_loader, valid_data_loader, test_data_loader = get_train_val_test_data(a, labels, co_fn='collate2')
        # train_data_loader = get_pre_data(a, labels, co_fn='collate2')


    else:
        a = list(zip(all_feature_m, all_adj_m, all_com_feature))
        # labels = normalizer(np.array(labels).reshape(-1, 1)).absmax_normalize()   # NbSi数据做归一化处理
        # labels = np.array(labels) + 1 # Nb数据负值预测结果不好，给每个label加上一个固 定值
        train_data_loader, valid_data_loader, test_data_loader = get_train_val_test_data(a, labels, co_fn='collate2')
        # train_data_loader = get_pre_data(a, labels, co_fn='collate2')

    return train_data_loader, valid_data_loader, test_data_loader
    # return train_data_loader
