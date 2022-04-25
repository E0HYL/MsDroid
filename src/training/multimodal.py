#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2020/07/05 22:07:45
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2020
@Desc    :   Define loss, build model, enable train and test. 
'''

# here put the import lib
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.data import DataLoader
from torch_geometric.data import Dataset
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import torch.optim as optim
import time
from datetime import datetime
from tqdm import tqdm
import os
import os.path as osp
import numpy as np
from tensorboardX import SummaryWriter
import logging

from training.loader import real_batch, GraphDroidDataset
from utils import get_device, fscore, metric2scores


flens = [268, 224]
        
def disable_feature(x, key):
    seps = [0, flens[0]]
    for i in range(len(flens)-1):
        seps.append(seps[i+1]+flens[i+1])
    sep_features = [x[:, seps[i]:seps[i+1]] for i in range(len(seps)-1)]
    
    if key == 2:
        feature = torch.ones(sep_features[0].shape[0], 1)
    elif key in [0, 1]:
        feature = sep_features[key]
    else:
        raise ValueError('Disable feature `key` options. 0: keep permission, 1: keep opcodes, 2: disable both.')

    return feature


class GNNStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, conv_func=None, global_pool=None, train_eps=False, layer_norm=False, mask=None):
        super(GNNStack, self).__init__()
        self.convs = nn.ModuleList()
        self.conv_func = conv_func
        self.train_eps = train_eps
        self.mask = mask
        self.convs.append(self.build_conv_model(input_dim, hidden_dim))
        self.norm = nn.ModuleList()
        if layer_norm:
            self.norm.append(nn.LayerNorm(hidden_dim))
            self.norm.append(nn.LayerNorm(hidden_dim))
        else:
            self.norm.append(pyg_nn.BatchNorm(hidden_dim))
            self.norm.append(pyg_nn.BatchNorm(hidden_dim))

        for l in range(2):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))
        self.global_pool = global_pool

        # post-message-passing
        if self.global_pool == 'mix':
            self.post_mp = nn.Sequential(
                # -> CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->??                # nn.Linear(hidden_dim*2, hidden_dim*2), nn.ReLU(inplace=True), # mix_relu
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
        if self.mask is not None:
            x = disable_feature(x, self.mask)
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

        emb = x
        x = self.post_mp(x)
        if self.mask is None:
            out = F.log_softmax(x, dim=1)
        else:
            emb = x
            out = None
        return emb, out
    
    
def fine_tune(loader, test_loader, per_dict, opc_dict, dev=None, batch_size=64, num_epoch=100, start_epoch=0, dimension=128):
    dev = get_device(dev)
    logger.info('Starting Fine Tuning')
    model = multi_GNNStack(per_dict, opc_dict, dimension).to(dev)
    best = [0, 0, 0, 0, 0]
    
    opt = optim.Adam(model.parameters(), lr=0.001)
    min_loss = loss_model = None
    best_model = {i:None for i in range (5)}
    flag = False
    
    writer = SummaryWriter('tensorboard')
    try:
        # train
        for epoch in range(num_epoch):
            total_loss = 0
            model.train()
            
            T1 = time.process_time()
            for batch in tqdm(loader, desc=f'Epoch {epoch}'):
                opt.zero_grad()
                batch, position = real_batch(batch)
                
                # logger.info('batch traing for %d subgraphs' % len(batch.y))
                embedding, pred = model(batch.to(dev))
                label = batch.y
                loss = model.apk_loss(pred, label, position)
        
                loss.backward()
                opt.step()
                total_loss += loss.item()
                torch.cuda.empty_cache()
            T2 = time.process_time()
            logger.info(f'[Timer] Epoch@{epoch}: {T2-T1}')
            
            del batch, label, embedding, pred, loss
            torch.cuda.empty_cache()
                
            total_loss /= len(loader.dataset) # mean loss of that epoch
            r_epoch = start_epoch + epoch

            precission, recall, accuracy = my_test(test_loader, model, dev)
            f1 = fscore(precission, recall, 1) # f1-score
            f2 = fscore(precission, recall, 2)
            logger.info("Epoch {}. Loss: {:.4f}. [TEST] precission: {:.4f} recall: {:.4f} accuracy: {:.4f}, F1: {:.4f}".format(
                r_epoch, total_loss, precission, recall, accuracy, f1))
            if epoch % 5 == 0:
                writer.add_scalar("Train_Loss", total_loss, r_epoch)
                writer.add_scalar("Test_Precission", precission, r_epoch)
                writer.add_scalar("Test_Recall", recall, r_epoch)
                writer.add_scalar("Test_Accuracy", accuracy, r_epoch)
                writer.add_scalar("Test_F1-score", f1, r_epoch)
                writer.add_scalar("Test_F2-score", f2, r_epoch)

            if r_epoch > 500 and not sum(best):
                flag = True
            if f1 > 0.95 or (flag and f1 > 0.92):
                store = [precission, recall, accuracy, f1, f2]
                savename = '%f_%f_%f_%f_%f' % (precission, recall, accuracy, f1, f2)
                # save best precission or recall or accuracy or f1 or f2 model
                for i in range(5):
                    if store[i] > best[i]:
                        best[i] = store[i]
                        torch.save(model.state_dict(), savename)
                        tmp = best_model[i]
                        best_model[i] = savename
                        if tmp is not None:
                            if osp.exists(tmp) and tmp not in best_model.values():
                                os.remove(tmp)
                # save min train loss model (if not in best models)
                if min_loss is None:
                    min_loss = total_loss
                elif total_loss < min_loss:
                    min_loss = total_loss
                    if savename not in best_model.values():
                        if loss_model is not None:
                            os.remove(loss_model)
                        torch.save(model.state_dict(), savename)
                        loss_model = savename

    except Exception:
        logger.exception(f'Exception while training batch `{batch}` in No.{epoch} epoch.')
        epoch -= 1
    finally:
        return epoch+1, model


def my_test(loader, model, dev=None, is_validation=False, curve=False):
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
    if is_validation:
        api_preds = []
    if curve:
        apk_labels = []
        apk_preds = []
        apk_plabel = []
    
    TP = TN = FN = FP = 0
    for data in loader:
        data, position = real_batch(data)
        with torch.no_grad():
            emb, pred = model(data.to(dev))
            if curve:
                pred_score = pred[:,1]
            pred = pred.argmax(dim=1) # 0 or 1
            label = data.y
            if is_validation:
                api_preds += pred.tolist() # api_labels in a batch
                continue
            
        for i in range(len(position)-1):
            start, end = position[i:i+2]
            apk_pred = pred[start:end]
            apk_label = label[start:end]
            unilabel = set(apk_label.tolist())
            
            assert len(unilabel)==1
            unilabel = list(unilabel)[0]
            apk_pred = apk_pred.sum().sign().item()
            # logger.info("Label: %d \t Prediction:%s" % (unilabel, apk_pred))
            if curve:
                apk_pred_score = pred_score[start:end]
                apk_preds.append(apk_pred_score.max().item())
                apk_plabel.append(apk_pred)
                apk_labels.append(unilabel)
            else:          
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
                    
    if is_validation:
        return api_preds
    if curve:
        return apk_preds, apk_labels, apk_plabel
    else:
        precission, recall, accuracy = metric2scores(TP, FP, TN, FN, f=False)   
        return precission, recall, accuracy


class multi_GNNStack(nn.Module):
    def __init__(self, per_dict, opc_dict, dimension, output_dim=2, weights=[0.75, 0.25]):
        super(multi_GNNStack, self).__init__()
        self.dimension = dimension
        self.model_per = GNNStack(flens[0], self.dimension, 2, global_pool=global_pool, layer_norm=layer_norm, mask=0)
        self.model_opc = GNNStack(flens[1], self.dimension, 2, global_pool=global_pool, layer_norm=layer_norm, mask=1)
        self.model_per.load_state_dict(torch.load(per_dict))
        self.model_opc.load_state_dict(torch.load(opc_dict))
        flag = 0
        for m in [self.model_per, self.model_opc]:
            logger.info('Freezing')
            for param in m.parameters():
                param.requires_grad = False
            m.post_mp = nn.Linear(m.post_mp[0].in_features, int(self.dimension*weights[flag]))
            flag += 1
        
        self.linear = nn.Linear(int(self.dimension*(sum(weights))), output_dim)

    def forward(self, data):
        per_emb, _ = self.model_per(data)
        opc_emb, _ = self.model_opc(data)
        out = torch.cat((per_emb, opc_emb), 1)
        x = self.linear(out)
        x = F.log_softmax(x, dim=1)
        return out, x

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
                # logger.info('Benign Loss: %f' % apk_loss.item())
            else:
                scores = []
                for j in range(end-start):
                    scores.append(F.nll_loss(apk_pred[j:j+1], apk_label[j:j+1]))
                apk_loss = min(scores)
                # logger.info('Malware Loss: %f' % apk_loss.item())
            
            loss += apk_loss
        return loss


def torch_loader(data_path, shuffle=True, batch_size=64):
    data = torch.load(data_path)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    return loader


def find_best_model(keys=['f1'], model_path='.'):
    import glob
    import pandas as pd
    models = glob.glob(f'{model_path}/*')
    scores = []
    for m in models:
        dict_name = m.split('/')[-1]
        if not dict_name.startswith('last_epoch_'):
            m_score = [m]
            m_score.extend(dict_name.split('_'))
            scores.append(m_score)
    scores = pd.DataFrame(scores, columns=['model', 'precission', 'recall', 'accuracy', 'f1', 'f2'])
    selected = []
    for k in keys:
        values = scores[k]
        indexes = values[values==values.max()].index
        m0 = scores['model'][indexes].tolist()
        selected.extend(m0)
    return selected # list


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Multi-Model Fine Tuning')
    parser.add_argument('--node', '-n', help='node number of dummy input', type=int, default=35)
    parser.add_argument('--edge', '-e', help='edge number of dummy input', type=int, default=122)
    parser.add_argument('--feature', '-f', help='feature dimension of dummy input', type=int, default=492)
    parser.add_argument('--dimension', '-d', help='hidden layer embedding dimension', type=int, default=128)
    parser.add_argument('--pool', '-p', help='global pooling function', default='mix')
    parser.add_argument('--layer', '-l', help='set True if you want LayerNorm, else use BatchNorm', default=False)
    parser.add_argument('--dev', '-v', help='gpu device id', type=int, default=0)
    args = parser.parse_args()

    # set up logging to file 
    logging.basicConfig( 
        filename='fine_tune.log',
        level=logging.INFO, 
        format='[%(asctime)s]{%(pathname)s:%(lineno)d}%(levelname)s- %(message)s', 
        datefmt='%H:%M:%S'
    ) 
    # set up logging to console 
    console = logging.StreamHandler() 
    console.setLevel(logging.DEBUG) 
    # set a format which is simpler for console use 
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter) 
    # add the handler to the root logger 
    logging.getLogger('').addHandler(console) 
    logger = logging.getLogger(__name__)

    num_edge, num_node, num_node_features = [args.edge, args.node, args.feature]
    print(f'[INFO] num_node: {num_node}, num_edge: {num_edge}, num_node_features: {num_node_features}')
    data = Data(x=torch.randn(num_node, num_node_features), edge_index=torch.LongTensor(2*num_edge).random_(0, num_node).reshape(2, num_edge))

    dimension, global_pool, layer_norm = [args.dimension, args.pool, args.layer]
    print(f'[INFO] embedding dimension: {dimension}\n')
    # model = GNNStack(num_node_features, dimension, 2, global_pool=global_pool, layer_norm=layer_norm)
    
    """
    Set three following paths first.
    """
    # per_dict = 
    # opc_dict = 
    # data_dir = 
    
    model = multi_GNNStack(per_dict, opc_dict, dimension)
    data = [data]
    for dummy_input in DataLoader(data, batch_size=len(data)):
        print(model(dummy_input)[1].shape)
        macs, params = profile(model, inputs=(dummy_input, ))
        macs, params = clever_format([macs, params], "%.3f")
        print(f'\n[INFO] macs: {macs}, params: {params}')
    print(f'[INFO] model structure: \n{model}')

    logger.info(f'Loading train and test from {data_dir}.')
    fine_tune(torch_loader(f'{data_dir}train.pt'), torch_loader(f'{data_dir}test.pt'), per_dict, opc_dict, dev=args.dev)
