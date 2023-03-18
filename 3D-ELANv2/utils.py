import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import torch as th
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F

device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')

def draw_epoches(epoch_losses,title=''):
  plt.xlabel('epoches', fontsize=10)
  plt.ylabel('Train loss', fontsize=10)
  plt.plot(list(range(1,len(epoch_losses)+1)),epoch_losses)
  plt.title(title)
  plt.show()

def draw_regression_curve(test_labels,test_preds,score):
    plt.figure(figsize=(10, 6))
    plt.xlabel('data_id', fontsize=10)
    plt.ylabel('energy', fontsize=10)
    plt.subplot(2,1,1)
    plt.plot(range(len(test_preds)),test_preds, marker='o',color='b',label='prediction')
    plt.plot(range(len(test_preds)),test_labels,marker='x',color='r',label='target')
    plt.title('regression,r2_score:{:.3f},mse:{:.3f},mae:{:.3f}'.format(*score))
    plt.legend()
    plt.grid()
    plt.subplot(2,1,2)
    plt.scatter(test_labels,test_preds)
    plt.plot(range(-10,10),range(-10,10))
    plt.grid()
    plt.show()

def accuracy(output, y):
    return (output.argmax(1) == y).type(th.float32).mean().item()

def evaluate(model,features,adj,labels):
    model.eval()
    output = model(features, adj)
    loss = F.mse_loss(output, labels).item()
    return loss