#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   explainer.py
@Time    :   2020/07/05 22:04:06
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2020
@Desc    :   GNNExplainer for graph classification.
'''

# here put the import lib
from math import sqrt
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_networkx
import numpy as np
from sklearn.preprocessing import normalize
import os

from graph import plot_api_subgraph
EPS = 1e-15


class GraphExplainer(torch.nn.Module):
    r"""
    Args:
        model (torch.nn.Module): The GNN module to explain.
        epochs (int, optional): The number of epochs to train.
            (default: :obj:`100`)
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.01`)
        log (bool, optional): If set to :obj:`False`, will not log any learning
            progress. (default: :obj:`True`)
    """

    coeffs = {
        'edge_size': 0.001,
        'non_size': 0.5,
        'edge_ent': 1.0,
    }

    def __init__(self, model, epochs=100, lr=0.01, log=True, node=False): # disable node_feat_mask by default
        super(GraphExplainer, self).__init__()
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.log = log
        self.node = node

    def __set_masks__(self, x, edge_index, init="normal"):
        (N, F), E = x.size(), edge_index.size(1)

        std = 0.1
        if self.node:
            self.node_feat_mask = torch.nn.Parameter(torch.randn(F) * 0.1)

        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        self.edge_mask = torch.nn.Parameter(torch.randn(E) * std)
        self.edge_mask = torch.nn.Parameter(torch.zeros(E)*50)
        
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask
        
    def __clear_masks__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
        if self.node:
            self.node_feat_masks = None
        self.edge_mask = None

    def __num_hops__(self):
        num_hops = 0
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                num_hops += 1
        return num_hops

    def __flow__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'

    def __subgraph__(self, node_idx, x, edge_index, **kwargs):
        # num_nodes, num_edges = x.size(0), edge_index.size(1)
        x=x
        edge_index = edge_index
        row, _ = edge_index
        edge_mask = row.new_empty(row.size(0), dtype=torch.bool)
        edge_mask[:]=True
        return x, edge_index, edge_mask

    def __graph_loss__(self, log_logits, pred_label, non_edge):
        loss = -log_logits[0,pred_label]
        m = self.edge_mask.sigmoid()
        loss += self.coeffs['edge_size'] * m.sum() # penalize large size of the explanation
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m+ EPS)
        loss += self.coeffs['edge_ent'] * ent.mean() # encourage structural feature mask to be discrete
        m_non = m[non_edge]
        loss += self.coeffs['non_size'] * m_non.sum()
        
        return loss 

    def visualize_api_subgraph(self, data, edge_mask):
        edge_index = data.edge_index
        center = data.center.item()
        num_nodes = len(data.x)
        types = data.labels[0]
        types = {i:types[i] for i in range(num_nodes)}
        mapping = data.mapping[0]
        mapping = {i:mapping[i].split('(')[0].split('/')[-1] for i in range(num_nodes)} 

        edge_color = edge_mask
        data = Data(edge_index=edge_index, num_nodes=num_nodes, att=edge_color)
        G = to_networkx(data, edge_attrs=['att'])

        
        """compare with common visualization"""
        # pos = nx.circular_layout(G)
        # fig, ax = plt.subplots()
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        # nx.draw(G, pos=pos)
        # nx.draw_networkx_labels(G, pos=pos)
        # # nx.draw_networkx_labels(G, pos=pos, labels=mapping, font_size=8)
        # plt.show()

        plot_api_subgraph(G, types, center, edge_cmap=True, edge_color=edge_color, labels=mapping)

    def explain_graph(self, data, **kwargs):
        self.model.eval()
        self.__clear_masks__()
        x, edge_index = data.x, data.edge_index
        
        types = data.labels[0]
        non_nodes = []
        node_id = 0
        for i in types:
            if i == 0:
                non_nodes.append(node_id)
            node_id += 1
        # print('NON NODE', non_nodes)
        e_id = 0
        non_edges = []
        for end_node in edge_index[1]:
            if end_node in non_nodes:
                non_edges.append(e_id)
            e_id += 1
        # print('NON EDGE', non_edges)

        x, edge_index, _ = self.__subgraph__(node_idx=None,x=x, edge_index=edge_index)
        # Get the initial prediction.
        with torch.no_grad():
            _, probs_Y = self.model(data)
            pred_label = probs_Y.argmax(dim=-1)
            kwargs['useclass'] = kwargs.get('useclass') or None
            if kwargs['useclass'] is None:
                print('[Model Prediction] %d' % pred_label)
            else:
                if not pred_label==kwargs['useclass']:
                    return None, None

        self.__set_masks__(x, edge_index)
        self.to(x.device)

        if self.node:
            optimizer = torch.optim.Adam([self.node_feat_mask, self.edge_mask],
                                         lr=self.lr)
        else:
            optimizer = torch.optim.Adam([self.edge_mask], lr=self.lr)
        
        epoch_losses=[]
        for epoch in range(1, self.epochs + 1):
            epoch_loss=0
            optimizer.zero_grad()
            _, pred = self.model(data)
            loss = self.__graph_loss__(pred, pred_label, non_edges)
            loss.backward()

            optimizer.step()
            epoch_loss += loss.detach().item()
            epoch_losses.append(epoch_loss)

        edge_mask = self.edge_mask.detach().sigmoid()

        self.__clear_masks__()

        return edge_mask, epoch_losses
    
    def __repr__(self):
        return f'{self.__class__.__name__}()'