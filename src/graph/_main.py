#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   _main.py
@Time    :   2020/07/20 14:30:49
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2020
@Desc    :   None
'''

# here put the import lib
import os.path as osp
from graph.behavior_subgraph import MyOwnDataset


def generate_graph(call_graphs, output_dir, apk_base, db_name, label, hop=2, tpl=True, training=False, api_map=False):
    exp_dir = f'./training/Graphs/{db_name}/HOP_{hop}/TPL_{tpl}' if training else osp.join(output_dir, db_name) 
    MyOwnDataset(root=exp_dir, label=label, tpl=tpl, hop=hop, db=db_name, base_dir=apk_base, apks=call_graphs, api_map=api_map)
