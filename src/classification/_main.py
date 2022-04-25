#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   _main.py
@Time    :   2020/09/22 23:45:48
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2020
@Desc    :   None
'''

# here put the import lib
import sys
import glob
import os.path as osp
import pandas as pd
from tqdm import tqdm
import torch
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from classification.model_test import my_test, MsDroid


def form_dataset(exp_dir, graph_num, save=False):
    db_path = f"{exp_dir}/processed"
    save_path = f'{db_path}/dataset.pt'
    if osp.exists(save_path):
        dataset = torch.load(save_path)
    else:
        dataset = []
        apk_ids = []
        for i in tqdm(range(graph_num), desc='dataset forming'):
            apk_subgraphs = glob.glob(f'{db_path}/data_{i}_*.pt')
            if len(apk_subgraphs):
                apk_ids.append(i)
                data_list = []
                for j in apk_subgraphs:
                    data = torch.load(j)
                    data_list.append(data)
                dataset.append(Data(data=data_list))
    if save: torch.save(dataset, save_path)
    return apk_ids, dataset


def classify(call_graphs, exp_dir, batch_size, device, model_path):
    apk_ids, dataset = form_dataset(exp_dir, len(call_graphs))
    test_loader = DataLoader(dataset, batch_size=batch_size)
    model = MsDroid(492, 128, 2, global_pool='mix').to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)), strict=False)
    preds, _, _, _ = my_test(test_loader, model, device, validate=False)
    
    apk_map = pd.DataFrame(call_graphs, columns=['apk']).reset_index()
    prediction = pd.DataFrame(columns=['index','pred'])
    prediction['index'] = apk_ids
    prediction['pred'] = preds
    pred_result = pd.merge(apk_map, prediction)
    pred_result.to_csv(f'{exp_dir}/prediction.csv', index=False)
    print(f'Prediction result saved in {exp_dir}/prediction.csv')

    # return pred_result


if __name__ == '__main__':
    current_path = sys.path[0]

    # dummy dataset
    num_apk = 1
    num_subgraph = 5
    num_edge, num_node, num_node_features = [122, 35, 492]
    # apk_ids = list(range(num_apk))
    dataset = []
    for i in range(num_apk):
        data_list = []
        for j in range(num_subgraph):
            data = Data(x=torch.randn(num_node, num_node_features), edge_index=torch.LongTensor(2*num_edge).random_(0, num_node).reshape(2, num_edge))
            data_list.append(data)
        dataset.append(Data(data=data_list))

    device = 'cpu'
    use_cuda = False if device == 'cpu' else True
    test_loader = DataLoader(dataset, batch_size=16)
    model = MsDroid(492, 128, 2, global_pool='mix').to(device)
    model.load_state_dict(torch.load(f'{current_path}/model.pkl'))
    with torch.autograd.profiler.profile(use_cuda=use_cuda) as prof:
        preds, _, _, _ = my_test(test_loader, model, device, validate=False)
    print('prediction:', preds)
    with open(f'{current_path}/overhead.txt', 'w') as f:
        f.write(str(prof))
