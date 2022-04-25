#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   form_dataset.py
@Time    :   2020/12/26 20:51:30
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2020
@Desc    :   None
'''

# here put the import lib
import glob
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import os


def form_dataset(db, hop, tpl, max_gid, map_path=None):
    db_path = f"./training/Graphs/{db}/HOP_{hop}/TPL_{tpl}/processed"
    print(db_path)
    if map_path is None:
        savename = 'dataset'
        map_path = f"./training/Mappings/{db}_{hop}_{tpl}.csv"
    else:
        savename = map_path.split('/')[-1]

    dataset = mapping_dataset(map_path, db_path, max_gid)
    torch.save(dataset, f'{db_path[:-10]}/{savename}.pt')


def mapping_dataset(map_path, db_path, max_gid):
    graph_num = max_gid + 1
    dataset = []
    names = ['graph_id','subgraph_id','apk','api']
    mappings = pd.DataFrame(columns=names)

    pbar = tqdm(range(graph_num))
    for graph_id in pbar:
        sub_list = [s[:-3].split('_')[-1] for s in glob.glob(f"{db_path}/data_{graph_id}_*.pt")]
        if len(sub_list):
            data_list = []
            for subgraph_id in sub_list:
                ptfile = f"{db_path}/data_{graph_id}_{subgraph_id}.pt"
                # dataset
                data = torch.load(ptfile)
                data_list.append(data)
                # mappings
                apk_name = data.app
                api_name = data.mapping[data.center]
                item = pd.DataFrame([[graph_id, subgraph_id, apk_name, api_name]], columns=names)
                mappings = mappings.append(item)
                pbar.set_description(ptfile.split('/')[-1])
            dataset.append(Data(data=data_list))
    
    mappings.to_csv(map_path, index=False)
    return dataset


def judge_(datasetpath):
    if not os.path.exists(datasetpath):
        return False
    elif os.path.getsize(datasetpath)<=149:
        return False
    else:
        return True


if __name__ == "__main__":
    db = "amd"; hop = 2; tpl = True; map_path = None
    form_dataset(db, hop, tpl, map_path)
