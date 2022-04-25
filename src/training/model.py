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
import logging
model_logger = logging.getLogger()

from training.loader import real_batch
from utils import get_device, fscore, metric2scores


class GNNStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, conv_func=None, global_pool=None, train_eps=False, layer_norm=False):
        super(GNNStack, self).__init__()
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

    def apk_hard_loss(self, pred, label, position, weights=True):
        loss = 0
        for i in range(len(position)-1):
            start, end = position[i:i+2]
            apk_pred = pred[start:end]
            apk_label = label[start:end]
            unilabel = set(apk_label.tolist())
            
            assert len(unilabel)==1
            unilabel = list(unilabel)[0]
            if not unilabel: # Benign
                apk_loss = F.nll_loss(apk_pred, apk_label)
                # print('Benign Loss: %f' % apk_loss.item())
            else:
                scores = []
                all_scores = []
                for j in range(end-start):
                    single_pred = apk_pred[j:j+1]
                    single_loss = F.nll_loss(apk_pred[j:j+1], apk_label[j:j+1])
                    all_scores.append(single_loss)
                    if single_pred.argmax(dim=1):
                        scores.append(single_loss)
                sclen = len(scores)
                if sclen:
                    if weights:
                        w = np.linspace(0, 1, num=sclen+1)
                        w = (w / sum(w))[1:]
                        scores.sort(reverse=True) # descending order(larger loss, smaller weight??                        apk_loss = 0
                        for i in range(len(w)):
                            apk_loss += scores[i]*w[i]  
                    else:
                        apk_loss = sum(scores) / len(scores)
                else:
                    apk_loss =  min(all_scores)
                # print('Malware Loss: %f' % apk_loss.item())
            
            loss += apk_loss
        return loss
    
    
def my_train(loader, test_loader, writer, model_dict, dev=None, lossfunc=0, batch_size=64, num_epoch=1000, start_epoch=0, best=None, conv_func=None, global_pool=None, train_eps=False, dimension=128, layer_norm=False):
    dev = get_device(dev)
    model_logger.info('Starting Training')
    # build model
    num_classes = 2
    num_node_features = loader.dataset[0].data[0].x.shape[1]
    model = GNNStack(num_node_features, dimension, num_classes, conv_func=conv_func, global_pool=global_pool, train_eps=train_eps, layer_norm=layer_norm).to(dev)

    dict_name = model_dict.split('/')[-1]
    if best is None:
        if dict_name.startswith('last_epoch_') or (dict_name == '0'):
            best = [0, 0, 0, 0, 0]
        else:
            best = [float(i) for i in dict_name.split('_')]
    if osp.exists(model_dict):
        model.load_state_dict(torch.load(model_dict))
    model_path = '/'.join(model_dict.split('/')[:-1])
    
    opt = optim.Adam(model.parameters(), lr=0.001)
    min_loss = loss_model = None
    best_model = {i:None for i in range (5)}
    flag = False
    
    try:
        # train
        for epoch in range(num_epoch):
            total_loss = 0
            model.train()
            
            T1 = time.process_time()
            for batch in tqdm(loader, desc=f'Epoch {epoch}'):
                opt.zero_grad()
                batch, position = real_batch(batch)
                
                # print('batch traing for %d subgraphs' % len(batch.y))
                embedding, pred = model(batch.to(dev))
                label = batch.y
                if lossfunc == 0:
                    loss = model.apk_loss(pred, label, position)
                elif lossfunc == 1:
                    loss = model.apk_hard_loss(pred, label, position)
                elif lossfunc == 2:
                    loss = model.apk_hard_loss(pred, label, position, weights=False)
        
                loss.backward()
                opt.step()
                total_loss += loss.item()
                torch.cuda.empty_cache()
            T2 = time.process_time()
            model_logger.info(f'[Timer] Epoch@{epoch}: {T2-T1}')
            
            del batch, label, embedding, pred, loss
            torch.cuda.empty_cache()
                
            total_loss /= len(loader.dataset) # mean loss of that epoch
            r_epoch = start_epoch + epoch

            precission, recall, accuracy = my_test(test_loader, model, dev)
            f1 = fscore(precission, recall, 1) # f1-score
            f2 = fscore(precission, recall, 2)
            model_logger.info("Epoch {}. Loss: {:.4f}. [TEST] precission: {:.4f} recall: {:.4f} accuracy: {:.4f}, F1: {:.4f}".format(
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
            if f1 > 0.95 or (flag and f1 > 0.85):
                store = [precission, recall, accuracy, f1, f2]
                savename = '%s/%f_%f_%f_%f_%f' % (model_path, precission, recall, accuracy, f1, f2)
                # save best precission or recall or accuracy or f1 or f2 model
                for i in range(5):
                    if store[i] > best[i]:
                        best[i] = store[i]
                        if not osp.exists(savename):
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
                        
    except Exception as e:
        print(e)
        model_logger.exception(f'Exception while training batch `{batch}` in No.{epoch} epoch.')
        epoch -= 1
    finally:
        return epoch+1, model


def my_test(loader, model, dev=None, is_validation=False, curve=False, emb_=False):
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
    if emb_:
        embeddings = []
    
    TP = TN = FN = FP = 0
    for data in loader:
        data, position = real_batch(data)
        with torch.no_grad():
            emb, pred = model(data.to(dev))
            if emb_:
                embeddings.extend(emb)
                continue
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
            # print("Label: %d \t Prediction:%s" % (unilabel, apk_pred))
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
    elif curve:
        return apk_preds, apk_labels, apk_plabel
    elif emb_:
        return embeddings
    else:
        precission, recall, accuracy = metric2scores(TP, FP, TN, FN, f=False)   
        return precission, recall, accuracy


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Model Efficiency Analysis')
    parser.add_argument('--node', '-n', help='node number of dummy input', type=int, default=35)
    parser.add_argument('--edge', '-e', help='edge number of dummy input', type=int, default=122)
    parser.add_argument('--feature', '-f', help='feature dimension of dummy input', type=int, default=492)
    parser.add_argument('--dimension', '-d', help='hidden layer embedding dimension', type=int, default=128)
    parser.add_argument('--pool', '-p', help='global pooling function', default='mix')
    parser.add_argument('--layer', '-l', help='set True if you want LayerNorm, else use BatchNorm', default=False)
    args = parser.parse_args()

    num_edge, num_node, num_node_features = [args.edge, args.node, args.feature]
    print(f'[INFO] num_node: {num_node}, num_edge: {num_edge}, num_node_features: {num_node_features}')
    data = Data(x=torch.randn(num_node, num_node_features), edge_index=torch.LongTensor(2*num_edge).random_(0, num_node).reshape(2, num_edge))

    dimension, global_pool, layer_norm = [args.dimension, args.pool, args.layer]
    print(f'[INFO] embedding dimension: {dimension}\n')
    model = GNNStack(num_node_features, dimension, 2, global_pool=global_pool, layer_norm=layer_norm)

    data = [data]
    from thop import profile, clever_format
    for dummy_input in DataLoader(data, batch_size=len(data)):
        macs, params = profile(model, inputs=(dummy_input, ))
        # macs, params = clever_format([macs, params], "%.3f")
        print(f'\n[INFO] macs: {macs}, params: {params}')
    # print(f'[INFO] model structure: \n{model}')
