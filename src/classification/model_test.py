#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   _model.py
@Time    :   2020/07/05 22:07:45
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2020
@Desc    :   Classification using trained model.
'''

# here put the import lib
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.data import Dataset
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import torch.optim as optim
from tqdm import tqdm
import os.path as osp
from utils import metric2scores, get_device


class MsDroid(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, conv_func=None, global_pool=None, train_eps=False, layer_norm=False):
        super(MsDroid, self).__init__()
        self.convs = nn.ModuleList()
        self.conv_func = conv_func
        self.train_eps = train_eps
        self.convs.append(self.build_conv_model(input_dim, hidden_dim))
        self.norm = nn.ModuleList()
        if layer_norm:
            self.norm.append(nn.LayerNorm(hidden_dim))
            self.norm.append(nn.LayerNorm(hidden_dim))
        else:
            self.norm.append(pyg_nn.BatchNorm(hidden_dim))
            self.norm.append(pyg_nn.BatchNorm(hidden_dim))

        for _ in range(2):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))
        self.global_pool = global_pool

        # post-message-passing
        if self.global_pool == 'mix':
            self.post_mp = nn.Sequential(
                # -> CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->�?                # nn.Linear(hidden_dim*2, hidden_dim*2), nn.ReLU(inplace=True), # mix_relu
                nn.Linear(hidden_dim*2, hidden_dim), nn.Dropout(0.25), 
                nn.Linear(hidden_dim, output_dim))
        else:
            self.post_mp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.25), 
                nn.Linear(hidden_dim, output_dim))

        self.dropout = 0.25
        self.num_layers = 3

    def build_conv_model(self, input_dim, hidden_dim):
        if not self.conv_func:
            return pyg_nn.GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                                nn.Linear(hidden_dim, hidden_dim)), train_eps=self.train_eps)
        elif self.conv_func == 'GATConv':
            return pyg_nn.GATConv(input_dim, hidden_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if data.num_node_features == 0:
            x = torch.ones(data.num_nodes, 1)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if not i == self.num_layers - 1:
                x = self.norm[i](x)

        if not self.global_pool:
            x = pyg_nn.global_mean_pool(x, batch)
        elif self.global_pool == 'max':
            x = pyg_nn.global_max_pool(x, batch)
        elif self.global_pool == 'mix':
            x1 = pyg_nn.global_mean_pool(x, batch)
            x2 = pyg_nn.global_max_pool(x, batch)
            x = torch.cat((x1, x2), 1)

        x = self.post_mp(x)
        emb = x
        out = F.log_softmax(x, dim=1)

        return emb, out
    
    def apk_loss(self, pred, label, position):
        loss = 0
        for i in range(len(position)-1):
            start, end = position[i:i+2]
            apk_pred = pred[start:end]
            apk_label = label[start:end]
            unilabel = set(apk_label.tolist())
            
            assert len(unilabel)==1
            unilabel = list(unilabel)[0]
            if not unilabel: # Benign
                apk_loss = F.nll_loss(apk_pred, apk_label)  # log_softmax + nll_loss => cross_entropy
                # print('Benign Loss: %f' % apk_loss.item())
            else:
                scores = []
                for j in range(end-start):
                    scores.append(F.nll_loss(apk_pred[j:j+1], apk_label[j:j+1]))
                apk_loss = min(scores)
                # print('Malware Loss: %f' % apk_loss.item())
            
            loss += apk_loss
        return loss

def my_test(loader, model, dev='cpu', validate=True, pred_api=False, pred_emd=False):
    """ confusion matrix 
    `prediction` and `truth`
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """
    model.eval()
    if dev is None:
        dev = get_device(dev)
        model.to(dev)
    metrics = None
    if validate:
        TP = TN = FN = FP = 0
    api_pred = []
    emd_pred = []
    apk_preds = []
    
    for data in tqdm(loader, desc='Batch Test'):
        data, position = real_batch(data)
        with torch.no_grad():
            emb, pred = model(data.to(dev))
            pred = pred.argmax(dim=1) # 0 or 1
            if pred_api:
                api_preds += pred.tolist()
            if pred_emd:
                emd_pred += emb.tolist()
            
        for i in range(len(position)-1):
            start, end = position[i:i+2]
            apk_pred = pred[start:end]
            apk_pred = apk_pred.sum().sign().item()
            apk_preds.append(apk_pred)
            
            if validate:
                label = data.y
                apk_label = label[start:end]
                unilabel = set(apk_label.tolist())
                assert label != 2
                assert len(unilabel) == 1
                unilabel = list(unilabel)[0]    
                if apk_pred==unilabel:
                    if unilabel:
                        TP += 1
                    else:
                        TN += 1
                else:
                    if unilabel: # pred=0, label=1
                        FN += 1
                    else:
                        FP += 1
                    
    if validate:
        metrics = metric2scores(TP, FP, TN, FN)   # precission, recall, accuracy
    
    return apk_preds, metrics, api_pred, emd_pred


def real_batch(batch):
    '''
    Model would be generated for APIs using APK labels.
    Batch Trick: 
        Input Batch is generated for APKs because we don't want to seperate the APIs inside. So the real batch size is not fixed for each. `position` indicates boundaries for each APK inside the batch.
    '''
    real = []
    position = [0]
    count = 0
    for apk in batch.data:
        for api in apk:
            real.append(api)
        count += len(apk)
        position.append(count)
    real = DataLoader(real, batch_size=len(real))
    for r in real:
        '''
        one batch (batch_size=len(real))
        real_batch_size approximately equal to batch_size*avg(apk_subgraph_num)
        '''
        b = r
    return b, position

