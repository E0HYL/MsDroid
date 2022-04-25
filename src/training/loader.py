#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   loader.py
@Time    :   2020/07/05 22:05:51
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2020
@Desc    :   Dataset related operations.
'''

# here put the import lib
import glob
import os.path as osp
import os
import torch
import pandas as pd
from torch_geometric.data import Data, DataLoader

base_dir = f'./training/Graphs'


class GraphDroidDataset():
    def __init__(self, db, hop, tpl, base_dir=base_dir):
        self.db = db
        self.hop = hop
        self.tpl = tpl
        self.base_dir = base_dir
        self.data_base = get_base_name(self.db, self.hop, self.tpl, self.base_dir)
        if not os.path.exists(self.data_base):
            raise Exception(f'[DatasetNotExists] {self.data_base}')
        self.processed_dir = f'{self.data_base}/processed'
        self.get_feature_dim()


    def describe(self):
        desc = dataset_desc(self.db, self.base_dir)
        print(f'{desc[0]} APKs (graphs), {desc[1]} sensitive APIs (subgraphs);')
        print(f'APK subgraph number summary:\n{desc[2]}\n')
        return desc


    def load_apk(self, graph, mode=None, **kwargs):
        '''
        Both APK name and graph ID are supported.
        # Get the list of subgraphs data of the APK:
            dataset = GraphDroidDataset(db, hop, tpl, base_dir=base_dir)
            dataset.load_apk(graph).data 
        '''
        graph_id = name2id(self.db, graph, base_dir=self.base_dir)
        files = glob.glob(osp.join(self.processed_dir, f'data_{graph_id}_*'))
        subgraphs = [int(i.split('_')[-1].split('.')[0]) for i in files]
        sample = []
        for subgraph in subgraphs:
            data = self.load_api(graph, subgraph, mode=mode, **kwargs)
            sample.append(data)
        return Data(data=sample)


    def load_api(self, graph, subgraph, mode=None, **kwargs):
        '''
        Both APK/API name and graph/subgraph ID are supported.
        '''
        graph_id, subgraph_id = name2id(self.db, graph, api=subgraph, base_dir=self.base_dir)
        data = torch.load(osp.join(self.processed_dir, f'data_{graph_id}_{subgraph_id}.pt'))
        if mode is not None:
            data = self.__handle_api_x(data, mode, **kwargs)
        return data


    def load_db_apis(self, apiname, apk=True, mode=None, **kwargs):
        matched = self.api_search(apiname)
        apks = []
        apidata = []
        for m in matched:
            apks.append(matched[m]['apk'])
            graph_id = matched[m]['graph_id']
            if apk:
                apidata.append(self.load_apk(graph_id))
            else:
                subgraph_id = matched[m]['subgraph_id']
                apidata.append(self.load_api(graph_id, subgraph_id, mode=mode, **kwargs))
        return apidata, apks


    def get_mapping(self):
        mapping = get_db_mapping(f'{self.db}_{self.hop}_{self.tpl}', self.base_dir)
        x=pd.DataFrame()
        for i in range(mapping.graph_id.max() + 1):
            x=x.append(mapping[mapping.graph_id==i])
        return x


    def get_dataset(self, norm_opcode=False, mask=-1):
        if (not norm_opcode) and (mask == -1):
            return torch.load(osp.join(self.data_base, 'dataset.pt'))
        if norm_opcode:
            dataset = self.__norm_dataset_opcode()
        if mask != -1:
            dataset = self.__mask_dataset(mask)
        return dataset


    def __mask_dataset(self, mask=0):
        dataset = self.get_dataset()
        new_dataset = []
        for apk in dataset:
            apis = apk.data
            api_list = []
            for api in apis:
                data = self.__handle_api_x(api, 'mask', mask=mask)
                api_list.append(data)
            new_dataset.append(Data(data=api_list))
        return new_dataset


    def __norm_dataset_opcode(self):
        dataset = self.get_dataset()
        new_dataset = []
        for apk in dataset:
            apis = apk.data
            api_list = []
            for api in apis:
                data = self.__handle_api_x(api, 'norm')
                api_list.append(data)
            new_dataset.append(Data(data=api_list))
        return new_dataset
    

    def __handle_api_x(self, api, mode, **kwargs): # mode: 'mask' or 'norm'
        tmp = {api_attr[0]:api_attr[1] for api_attr in api}
        if mode == 'mask':
            kwargs['mask'] = kwargs.get('mask')
            x = disable_feature(api.x, self.data_base, kwargs['mask'])
        if mode == 'norm':
            x = norm_opcode(api.x, self.data_base)
        data = Data(x=x)
        for i in tmp:
            if i != 'x':
                data[i] = tmp[i]
        return data


    def get_feature_dim(self):
        self.flens = get_feature_len(self.db, self.hop, self.tpl, base_dir=self.base_dir)


    def get_feature_specific(self, fwords=['permission','opcodes']):
        dims = ' + '.join([str(l) for l in self.flens])
        print(f'Node FeatureVector {[fwords[i] for i in range(len(self.flens))]}: {dims}')
        return self.flens


    def api_search(self, apiname):
        mapping = self.get_mapping()
        apis = mapping[mapping.api==apiname]
        return apis[['graph_id','subgraph_id', 'apk']].to_dict('index')


def dataset_desc(db, base_dir=base_dir):
    '''
    Returns:
        Num of APKs (graphs), sensitive APIs (subgraphs);
        Discription of subgraph numbers for one apk (mean, std, min, max).
    '''
    mappings = get_db_mapping(db, base_dir)
    graph_ids = mappings.graph_id.value_counts().index.tolist()
    subgraphs = {}
    for i in graph_ids:
        subgraphs[i] = len(mappings[mappings.graph_id==i].subgraph_id)
    desc = pd.DataFrame(list(subgraphs.values())).describe()[0][1:]
    return len(graph_ids), len(mappings), desc


def api_search(apiname, db=None, base_dir=base_dir):
    if db is None:
        mapping_paths = glob.glob(f'{base_dir}/mappings/*.csv')
        m = pd.DataFrame()
        for i in mapping_paths:
            m = m.append(get_db_mapping(i, path=True))
    else:
        m = get_db_mapping(db, base_dir)
    apis = m[m.api==apiname]
    return apis[['graph_id','subgraph_id', 'apk']].to_dict('index')


def get_db_mapping(db, base_dir=base_dir, path=False):
    if not path:
        db = f'{base_dir}/mappings/{db}.csv'
    try:
        return pd.read_csv(db)
    except FileNotFoundError as identifier:
        raise FileNotFoundError(f'[MapNotExists]: {db}')


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


def name2id(db, apk, api=None, base_dir=base_dir):
    m = None
    if type(apk) == str or type(api) == str:
        m = get_db_mapping(db, base_dir)  
    graph_id = convert(m, apk, 'apk')
    if api is None:
        return graph_id

    subgraph_id = convert(m, api, 'api', graph_id)
    return graph_id, subgraph_id


def convert(mapping, value, keyword, apk_id=None):
    column = 'graph_id' if keyword == 'apk' else 'subgraph_id'
    if type(value) == int:
        graph_id = value
    elif type(value) == str:
        assert mapping is not None
        if keyword == 'api':
            assert apk_id is not None
            mapping = mapping[mapping.graph_id==apk_id]
        graph_id = set(mapping[mapping.apk==value][column])
        if len(graph_id) != 1:
            if keyword == 'api':
                value = f'{value}@{apk}'
            raise RuntimeError(f'{len(graph_id)} matching find for {value} in {db}')
        graph_id = list(graph_id)[0]
    else:
        raise TypeError(f'{type(value)} unsupported for `{keyword}`.')
    return graph_id 


def norm_opcode(x, exp_dir):
    import torch.nn.functional as F
    with open(f'{exp_dir}/FeatureLen.txt', 'r') as f:
        flens = eval(f.read())
        
    seps = [0, flens[0]]
    for i in range(len(flens)-1):
        seps.append(seps[i+1]+flens[i+1])
    sep_features = [x[:, seps[i]:seps[i+1]] for i in range(len(seps)-1)]
    
    fnum = 0
    for f in sep_features:
        if not fnum: # permission
            feature = f
        else:
            if fnum == 1: # opcode
                f = F.normalize(f, p=2, dim=0)
            feature = torch.cat([feature, f],1)
        fnum += 1
    return feature


def disable_feature(x, exp_dir, key):
    with open(f'{exp_dir}/FeatureLen.txt', 'r') as f:
        flens = eval(f.read())
        
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


def get_feature_len(db, hop, tpl, base_dir=base_dir):
    data_dir = get_base_name(db, hop, tpl, base_dir)
    try:
        with open(f'{data_dir}/FeatureLen.txt', 'r') as f:
            flens = eval(f.read())
    except FileNotFoundError:
        print('Feature dimension file not found.')
        flens = torch.load(glob.glob(f'{data_dir}/processed/data_*.pt')[0]).x.shape[1]
    return flens


def get_base_name(db, hop, tpl, base_dir=base_dir):
    data_base = f'{base_dir}/{db}/HOP_{hop}/TPL_{tpl}'
    return data_base
