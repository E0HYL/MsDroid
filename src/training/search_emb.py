#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   search_emb.py
@Time    :   2020/10/30 00:52:56
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2020
@Desc    :   None
'''

# here put the import lib
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Data
import pandas as pd
import os.path as osp
from tqdm import tqdm

from training.model import GNNStack, my_test
import logging
logger = logging.getLogger('main.emb')


def locate_api(api, mal_dbs=['amd-0', 'amd-1', 'amd-2'], map_base='./training/Mappings', hop=2, tpl=True):
    locations = {}
    apks = {}
    for i in mal_dbs:
        m = pd.read_csv(f'{map_base}/{i}_{hop}_{tpl}.csv')
        apks[i] = m[m.api==api].apk
        locations[i] = m[m.api==api].index
    return locations, apks


def api_similar_apk(api, e, dev=None, malemd='../helper/AMD_embeddings'):
    if not osp.exists(malemd):
        logger.error('Please generate your malware embedding db first!')
        return

    locations, apks = locate_api(api)
    apk = []
    apk_sim = []
    max_sim = 0

    for db in locations:
        l = locations[db]
        a = apks[db]
        fs = torch.load(f'{malemd}/{db}.pt')
        # print(len(fs))
        for i in tqdm(l, desc=db):
            f = fs[i]
            if dev is None:
                e.to(f.device)
            else:
                e.to(dev)
                f.to(dev)
            dist2 = F.pairwise_distance(torch.unsqueeze(e,0), torch.unsqueeze(f,0), p=2)

            sim = 1 / dist2 if dist2 != 0 else float('inf')
            if sim >= max_sim:
                apk_sim.append(sim)
                max_sim = sim
                apk.append((db, i, a[i]))

    results = []
    for i in range(len(apk_sim)):
        if apk_sim[i] == max_sim:
            results.append(apk[i])
    try:
        s = max_sim.item()
    except AttributeError:
        s = max_sim
    logger.info(f'Max similarity: {s}\n{results}')
    return s, results


def subgraph_cmp(subgraphs, model, dimension=128, dev=0, clever=True):
    results = []
    for data in subgraphs:
        api = data.mapping[data.center]
        logger.info(f'APK: {data.app}, API: {api}')
        test_loader = DataLoader([Data(data=[data])], batch_size=1)
        e = my_test(test_loader, model.to(dev), emb_=True, dev=dev)[0]
        s, r = api_similar_apk(api, e, dev)
        if clever:
            r = set(j[2].split('/')[0] for j in r)
        results.append([api, s, r])
    results = pd.DataFrame(results).sort_values(by=1,ascending=False)
    return results


if __name__ == "__main__":
    # apk_id = 136; api_id = 25
    # data = torch.load(f"/opt/data/E0/GraphDroid/GraphGenerator/Datasets/malware_vt5/HOP_2/TPL_True/processed/data_{apk_id}_{api_id}.pt") #data_136_25.pt
    # num_node_features = data.x.shape[1]
    # model = GNNStack(num_node_features, 128, 2, global_pool='mix')
    # model.load_state_dict(torch.load("/opt/data/E0/GraphDroid/GraphNN_bakup/experiments/20201021-232038/models/0.978533_0.967628_0.980562_0.973050_0.969789"))
    # print(subgraph_cmp([data], model))

    import logging
    # set up logging to file 
    logging.basicConfig( 
        filename='covid_scan.log',
        level=logging.INFO, 
        format='%(message)s', 
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

    import glob
    results = []
    model = None
    for i in range(160):
        logger.info(f'===================={i}====================')
        family = []
        apk_sim_list = []
        apis = glob.glob(f"/opt/data/E0/GraphDroid/GraphGenerator/Datasets/malware_vt5/HOP_2/TPL_True/processed/data_{i}_*.pt")
        for j in apis:
            data = torch.load(j)
            num_node_features = data.x.shape[1]
            if model is None:
                model = GNNStack(num_node_features, 128, 2, global_pool='mix')
                model.load_state_dict(torch.load("/opt/data/E0/GraphDroid/GraphNN_bakup/experiments/20201021-232038/models/0.978533_0.967628_0.980562_0.973050_0.969789"))
            api_sim_list = subgraph_cmp([data], model)[2][0]
            apk_sim_list.extend(list(api_sim_list))
        fa = pd.Series(apk_sim_list).value_counts().to_dict()
        results.append(str(fa))
    pd.Series(results).to_csv('malware_fam_cmp.csv', header=None)
