#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2020/09/22 15:43:36
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2020
@Desc    :   
'''

# here put the import lib
import sys
import os
from feature import generate_feature
from graph import generate_graph
from classification import classify
from utils import makedirs, set_logger, add_fh


def generate_behavior_subgraph(apk_base, db_name, output_dir, deepth, label, hop=2, tpl=True, training=False, api_map=False):
    '''
    <output_dir>/<db_name>/decompile/<apk_name>/call.gml
    <output_dir>/<db_name>/result/<permission | opcode | tpl>/<apk_name>.csv
    '''
    call_graphs = generate_feature(apk_base, db_name, output_dir, deepth)   # `.gml`
    call_graphs.sort()
    print("call graph", call_graphs)
    
    '''
    <output_dir>/<db_name>/processed/data_<apk_id>_<subgraph_id>.pt
    '''
    gml_base = f'{output_dir}/{db_name}'
    generate_graph(call_graphs, output_dir, gml_base, db_name, label, hop, tpl, training, api_map)
    return call_graphs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='MsDroid.')
    parser.add_argument('--input', '-i', help='APK directory')
    parser.add_argument('--output', '-o', help='output directory', default=f'{sys.path[0]}/Output')
    parser.add_argument('--device', '-d', help='device for model test', default='cpu')
    parser.add_argument('--batch', '-b', help='batch size for model test', default=16)
    parser.add_argument('--label', '-l', help='dataset label: malware(1) / benign(0), unnecessary if only prediction needed.', default=2)
    parser.add_argument('--deepth', '-dp', help='deepth of tpl seaching', default=3)
    args = parser.parse_args()

    input_dir = args.input
    apk_base = os.path.abspath(os.path.join(input_dir,'../'))
    db_name = input_dir.split(apk_base)[-1].strip('/')
    
    output_dir = args.output
    makedirs(output_dir)
    label = args.label
    
    import logging
    import time
    logger = logging.getLogger('main')
    set_logger(logger)
    add_fh(logger, 'main_log.log')

    T1 = time.process_time()    
    '''
    <output_dir>/<db_name>/processed/data_<apk_id>_<subgraph_id>.pt
    '''
    call_graphs = generate_behavior_subgraph(apk_base, db_name, output_dir, args.deepth, label)
    T2 = time.process_time()
    logger.info(f'[Timer] Generate Behavior Subgraphs for {len(call_graphs)} APKs: {T2-T1}')
    
    exp_dir = f'{output_dir}/{db_name}'
    T1 = time.process_time()
    classify(call_graphs, exp_dir, args.batch, args.device, f'{sys.path[0]}/classification/model.pkl')
    T2 = time.process_time()
    logger.info(f'[Timer] Classify {len(call_graphs)} APKs: {T2-T1}')
