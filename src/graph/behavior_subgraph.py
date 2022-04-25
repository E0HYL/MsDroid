#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   GraphGenerator.py
@Time    :   2020/07/05 16:34:22
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2020
@Desc    :   Main function to generate the (PyG format) dataset for GNN learning task.
'''

# here put the import lib
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
import torch
from torch_geometric.data import Dataset
import logging
import glob
import os
import os.path as osp
import time
from tqdm import tqdm

from graph.subgraph import api_subgraph
from utils import makedirs, find_all_apk

from functools import partial
from multiprocessing import Pool, cpu_count
import asyncio


class MyOwnDataset(Dataset):
    def __init__(self, root, tpl, hop, db, base_dir, transform=None, pre_transform=None, label=1, apks=None, layer=None, api_map=False):
        self.lens = 0
        self.samples = 0
        self.label = label
        self.base_dir = base_dir
        self.tpl = tpl
        self.hop = hop
        self.db = db
        self.apks = apks
        self.api_map = api_map
        if apks is None:
            self.layer = layer
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_paths(self):
        r"""The filepaths to find in order to skip the download."""
        if self.apks is None:
            db = self.db # self.root.split('/')[1]
            db_record = f'{self.base_dir}/{db}.csv'
            if not osp.isfile(db_record):
                print('[GraphDroid] Searching db for `.gml` files.')
                self.apks = get_db_gml(base_dir=self.base_dir, db=db, layer=self.layer)
                self.apks.to_csv(db_record, header=False, index=None)
            else:
                print(f'[GraphDroid] Read existing data csv: {db_record}')
                self.apks = pd.read_csv(db_record, header=None)[0]
        else:
            self.apks = pd.Series(self.apks)
        return self.apks

    @property
    def processed_file_names(self):
        r'''The name of the files to find in the :obj:`self.processed_dir`
        folder in order to skip the processing.'''
        ex_map = f'./mappings/{self.db}_2_True.csv'
        if osp.exists(ex_map):
            print(f'[GraphDroid] Read existing mapping csv: {ex_map}')
            df = pd.read_csv(ex_map)
            return [f'data_{v.graph_id}_{v.subgraph_id}.pt' for _,v in df.iterrows()]
        return [f'data_{i}_0.pt' for i in range(len(self.apks))]

    def _exclude_exists(self):
        print('[GraphDroid] Finding break points...')
        graph_ids = []
        apps = self.apks
        for i in tqdm(range(len(apps))):
            data_file = f'{self.root}/processed/data_{i}_0.pt'
            if osp.exists(data_file):
                a = torch.load(data_file).app
                # '''
                gml = apps[apps.str.contains(a)]
                assert len(gml) == 1
                apps = apps.drop(gml.index)
                # '''
                '''
                a_api = glob.glob(f'{self.root}/processed/data_{i}_*.pt')
                if len(pd.read_csv(apps[i].replace(f'{self.db}/decompile', f'{self.db}/result/permission').replace('/call.gml', '.csv'))) == len(a_api):
                    gml = apps[apps.str.contains(a)]
                    assert len(gml) == 1
                    apps = apps.drop(gml.index)
                    # print(f'[GraphDroid] Found {gml.item()}')
                else:
                    graph_ids.append(i)
                    for api in a_api:
                        os.remove(api)
                '''
            else:
                graph_ids.append(i)

        assert len(graph_ids) >= len(apps)
        return apps, graph_ids
    
    def _process(self):
        def files_exist(files):
            return len(files) != 0 and all([osp.exists(f) for f in files])
        apps = self.raw_paths
        if files_exist(self.processed_paths):
            print(f'[GraphDroid] Data found in `{self.root}/processed`. Skip processing.')
        else:
            if glob.glob(f'{self.root}/processed/data_*.pt'):
                apps, graph_ids = self._exclude_exists()
            else:
                graph_ids = list(range(len(apps)))
            makedirs(self.processed_dir)
            self.process(apps, graph_ids)
            if self.api_map:
                from graph.dbmap import form_dataset
                max_gid, _ = self.len()
                form_dataset(self.db, self.hop, self.tpl, max_gid)
            tqdm.write(f'[GraphDroid] Data generated in `{self.root}/processed/`.')

    def get_sep(self, example):
        fwords = ['permission','opcode']
        flens = get_feature(example, fwords=fwords, getsep=True, base_dir=self.base_dir)
        with open(f'{self.root}/FeatureLen.txt', 'w') as f:
            f.write(str(flens))

    def process(self, apps, graph_ids):
        self.get_sep(apps[0])
        apps = apps.sort_values()
        zip_args = list(zip(apps, graph_ids))
        logging.info(f'Processing {len(zip_args)} apps...')

        partial_func = partial(process_apk_wrapper, label=self.label, tpl=self.tpl, hop=self.hop, base_dir=self.base_dir, processed_dir=self.processed_dir) # fixed params
        self.samples, self.lens = mp_process(partial_func, zip_args)
        logging.info(f'Total app samples: {self.samples}, total behavior subgraphs: {self.lens}')

    def len(self):
        if not(self.samples & self.lens):
            pt_files = glob.glob(f"{self.processed_dir}/data_*.pt")
            self.lens = len(pt_files)
            gids = []
            for p in pt_files:
                gids.append(int(p.split('data_')[-1].split('_')[0]))
            self.samples = max(gids)
        return self.samples, self.lens

    def get(self, graph_id, subgraph_id):
        data = torch.load(osp.join(self.processed_dir, 'data_{}_{}.pt'.format(graph_id, subgraph_id)))
        return data


def mp_process(func, argument_list):
    num_pool = int(cpu_count() / 8)
    print('Number of pools:', num_pool)
    glen = 0
    slen = 0
    pool = Pool(processes=num_pool)
    jobs = [pool.apply_async(func=func, args=(*argument,)) if isinstance(argument, tuple) else pool.apply_async(func=func, args=(argument,)) for argument in argument_list]
    # https://stackoverflow.com/questions/38271547/when-should-we-call-multiprocessing-pool-join
    pool.close()
    T1 = time.process_time()
    for job in tqdm(jobs, desc='[GraphDroid GraphGenerator]'):
        gl, sl = job.get()
        glen += gl
        slen += sl
    pool.join()
    T2 = time.process_time()
    logging.info(f'[Timer] {T2-T1}')
    return glen, slen


def process_apk_wrapper(*args, **kwargs): # multiple arguments
    label = kwargs['label']
    tpl = kwargs['tpl']
    hop = kwargs['hop']
    base_dir = kwargs['base_dir']
    processed_dir = kwargs['processed_dir']
    app = args[0]
    graph_id = args[1]

    flag = 0
    num_subgraph = 0
    logging.info(app)
    try:
        data_list = asyncio.run(gml2Data(app, label, tpl=tpl, hop=hop, base_dir=base_dir))
        dlen = len(data_list)
        if dlen:
            for i in range(dlen):
                data = data_list[i]
                data_path = osp.join(processed_dir, 'data_{}_{}.pt'.format(graph_id,i))
                assert not osp.exists(data_path)
                torch.save(data, data_path)
                num_subgraph += 1
            flag = 1
        logging.info(f'[Success] {app}')
    except Exception:
        logging.exception(f'{app}')
    finally:
        return flag, num_subgraph


def get_feature(gmlfile, base_dir, fwords=['permission','opcode','tpl'], getsep=False):
    feature_file = gmlfile.replace(f'{base_dir}/decompile/', f'{base_dir}/result/%s/').replace('/call.gml', '.csv')
    print(feature_file)
    features = [feature_file % i for i in fwords]
    '''
    [node type]
        external: ('undefined'=0),'permission'=1
        'opcode'=2, 'tpl'=3
    '''
    if getsep:
        return [pd.read_csv(features[i]).shape[1]-1 for i in range(len(features))]
    return [pd.read_csv(features[i]).assign(type=i+1) for i in range(len(features))]


def convert_subgraph_edge(edge_index, feature_df, p, map_only=False):
    mapping = {int(row['id']):index for index,row in feature_df.iterrows()}
    center = mapping[p] # new
    if map_only:
        return mapping # (old, new)

    result=[]
    for l in edge_index:
        rep = [mapping[x] for x in l]
        result.append(rep)

    return result, center, mapping


async def prepare_file(gmlfile, base_dir, fwords):
    single_graph = nx.read_gml(gmlfile, label='id')
    x = get_feature(gmlfile, base_dir, fwords)
    return single_graph, x


async def generate_behavior_subgraph(p, features, single_graph, hop, debug, gmlfile, apk_name, y):
    nodes_type = features[['id', 'type']]
    subgraph_nodes, subgraph_edges, apimap = api_subgraph(p, single_graph, nodes_type, hop=hop, debug=debug)

    if len(subgraph_nodes) <= 1:
        logging.warning(f'[IsolateNode] {gmlfile}: isolated node@{p}')
        return None

    subtypes = nodes_type[nodes_type['id'].isin(subgraph_nodes)]
    subgraph_features = features[features.id.isin(subgraph_nodes)].reset_index(drop=True)
    assert subgraph_features.shape[0]==len(subgraph_nodes)

    edges = subgraph_edges # [(source, target, key), ...]
    edges_df = pd.DataFrame(edges).iloc[:,:-1].T

    edge_list, center, m = convert_subgraph_edge(edges_df.values.tolist(), subgraph_features, p)
    assert len(apimap)==len(m)
    mapping = [apimap[i] for i in m]
    labels = [subtypes[subtypes.id==i].type.tolist()[0] for i in m]

    data = Data(x=torch.tensor(subgraph_features.iloc[:,1:-1].values.tolist(), dtype=torch.float)
                , edge_index=torch.tensor(edge_list, dtype=torch.long)
                , y=torch.tensor([y], dtype=torch.long)
                , num_nodes=len(subgraph_nodes), labels=labels
                , center=center, mapping=mapping, app=apk_name)
    return data


async def gml2Data(gmlfile, y, base_dir, tpl=True, sub=True, hop=2, debug=False):
    fwords = ['permission','opcode','tpl'] if tpl else ['permission','opcode']
    single_graph, x = await prepare_file(gmlfile, base_dir, fwords)
    apk_name = gmlfile.split('/decompile/')[-1][:-9]
    all_nodes = pd.DataFrame(single_graph.nodes,columns=['id'])
    permission, opcodes = x[:2]
    if tpl:
        opcodes = pd.merge(opcodes, x[2], how='outer', on='id', suffixes=['_','']).drop(['type_'],axis=1)
        opcodes['type'] = opcodes['type'].fillna(2)
        opcodes = opcodes.fillna(0)
    features_exist = pd.merge(permission.astype('float'), opcodes, how='outer').fillna(0).drop_duplicates('id', keep='first')   # keep type = 1
    features = pd.merge(all_nodes, features_exist, how='outer').fillna(0)
    features['type'] = features['type'].astype('int')

    p_list = x[0].id.tolist()
    data_list = []
    if sub:
        tasks = []
        for p in p_list:
            partial_func = partial(generate_behavior_subgraph, features=features, single_graph=single_graph, hop=hop, debug=debug, gmlfile=gmlfile, apk_name=apk_name, y=y)
            tasks.append(partial_func(p))
        data_list = await asyncio.gather(*tasks)
        while None in data_list:
            data_list.remove(None)
    else:
        nodes = single_graph.nodes
        edges = single_graph.edges # [(source, target, key), ...]
        edge_list = pd.DataFrame(edges).iloc[:,:-1].T.values.tolist()
        data = Data(x=torch.tensor(features.iloc[:,1:].values.tolist(), dtype=torch.long)
                    , edge_index=torch.tensor(edge_list, dtype=torch.long)
                    , y=torch.tensor([y], dtype=torch.long) # y:list
                    , num_nodes=len(nodes))
        data_list.append(data)

    return data_list


def get_db_gml(base_dir, db='Drebin', check=False, layer=None):
    base = f'{base_dir}/{db}/'
    gmls = []
    if check:
        apks = find_all_apk(osp.join(base, db), end='.apk', layer=layer)
        for a in apks:
            rpath = a.split(db)[-1].split('.apk')[0]
            gmls.append(base+'decompile'+'%s/call.gml' % rpath)
        gmls = check_gml(gmls)
    else:
        gmls = find_all_apk(osp.join(base, 'decompile'), end='.gml', layer=layer)
    return pd.Series(gmls)


def check_gml(gmllist):
    tmp = []
    for a in gmllist:
        if not osp.exists(a):
            logging.warning(f'[NoGML] {a}')
        else:
            tmp.append(a)
    return tmp


if __name__ == '__main__':
    makedirs('loggings'); makedirs('mappings')

    import argparse
    parser = argparse.ArgumentParser(description='GraphDroid Data Generator.')
    parser.add_argument('db', type=str, help='Choose a decompiled APK dataset.')
    parser.add_argument('--tpl', type=str, default=True, help='Simpilfy third party library API nodes.')
    parser.add_argument('--hop', type=int, default=2, help='Subgraph based on k hop neighborhood.')
    parser.add_argument('--label', type=int, default=None, help='Dataset label: 1 for Malicious, 0 for Benign.')
    parser.add_argument('--base', type=str, default=None, help='Call graph and feature files directory.')
    parser.add_argument('--layer', type=int, default=1, help='Speed up gml searching.')
    args = parser.parse_args()

    LOG_FORMAT = '%(asctime)s %(filename)s[%(lineno)d] %(levelname)s - %(message)s'
    current_milli_time = lambda: int(round(time.time() * 1000))

    db = args.db
    tpl = args.tpl
    hop = args.hop

    exp_dir = f'./Datasets/{db}/HOP_{hop}/TPL_{tpl}'
    makedirs(exp_dir)

    logging.basicConfig(filename=f'./loggings/[HOP_{hop}-TPL_{tpl}-{db}]{current_milli_time()}.log', level=logging.INFO, format=LOG_FORMAT)
    logging.debug(exp_dir)

    if args.label is None:
        try:
            db_labels = {'Drebin':1, 'Genome':1, 'AMD':1, 'Benign':0}
            label = db_labels[db.split('_')[0]]
        except Exception:
            logging.error('Label must be specified for unkown dataset')
    else:
        label = args.label

    layer = None if args.layer < 0 else args.layer
    dataset = MyOwnDataset(root=exp_dir, label=label, tpl=tpl, hop=hop, db=db, base_dir=args.base, layer=layer)
