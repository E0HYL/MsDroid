#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   setup.py
@Time    :   2020/07/08 15:06:23
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2020
@Desc    :   Handle differently configurated experiments, the related stuffs include paths and trained models.
'''

# here put the import lib
import os
import torch
import glob
import shutil
import pandas as pd
from datetime import datetime
from torch_geometric.data import DataLoader
from tensorboardX import SummaryWriter

from training.model import GNNStack, my_test
from training.loader import GraphDroidDataset, get_feature_len
from utils import get_device, pr_curve, makedirs, search_pattern
import logging
setup_logger = logging.getLogger()

exp_base = './training/Experiments'
graph_base = './training/Graphs'

class Experiment():
    def __init__(self, trained_db, tpl, hop, norm_opcode, mask, model_config, exp_base=exp_base, graph_base=graph_base, force=False, continue_train=False, model_dict=None):
        self.trained_db = trained_db # set()
        self.tpl = tpl
        self.hop = hop
        self.norm_opcode = norm_opcode
        self.mask = mask
        self.model_config = model_config
        self.exp_id = None
        self.exp_base = exp_base
        self.graph_base = graph_base
        self.force = force
        self.continue_train = continue_train
        self.start_epoch = 0
        self.model_dict = model_dict
        self.__set_paths()

    def __set_paths(self):
        makedirs(self.exp_base)
        self.config_file = f'{self.exp_base}/exp_configs.csv'
        columns = ['exp_id', 'trained_db', 'tpl', 'hop', 'norm_opcode', 'mask', 'model_config']
        if not os.path.exists(self.config_file):
            (pd.DataFrame([], columns=columns)).to_csv(self.config_file, index=False)

        self.__get_exp_id()
        if self.exp_id is None or (self.force and not self.continue_train): # enable train mode: new exp / overwrite
            self.__create_exp(columns)
        else:
            setup_logger.info(f'Found an experiment with the configuration: {self.exp_id}')
            self.model_path = f'{self.exp_base}/{self.exp_id}/models'
            self.get_models(last=True, model_dict=self.model_dict)
            if self.continue_train: # enable train mode: continue
                if self.last_model.split('/')[-1] != '0':
                    clear = False
                    self.start_epoch = self.__epoch_from_log() + 1
                else:
                    clear = True
                    ori_log = f'{self.exp_base}/{self.exp_id}/exp_log.log'
                    if os.path.exists(ori_log):
                        os.remove(ori_log)
                self.__get_train_paths(clear=clear)
            else: # enable test mode
                # setup_logger.debug(f'Found models: {self.models}')
                setup_logger.debug(self.model_existence_notice())
                self.protest_path = f'{self.exp_base}/{self.exp_id}/portability'
                makedirs(self.protest_path)
        exp_data = f'{self.exp_base}/{self.exp_id}/TrainTest'
        makedirs(exp_data)
        self.exp_train = f'{exp_data}/train.pt'
        self.exp_test = f'{exp_data}/test.pt'
        self.log_path = f'{self.exp_base}/{self.exp_id}/exp_log.log'

    def __epoch_from_log(self):
        return int(search_pattern(f'{self.exp_base}/{self.exp_id}/exp_log.log', 'Epoch \\d+')[0].split(' ')[-1])
    
    def __create_exp(self, columns):
        setup_logger.info('Creating a new experiment.')
        old_exp_id = self.exp_id
        # 1. initialize the exp_id
        self.exp_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        # 2. modify exp_configs
        configs = pd.DataFrame([[self.exp_id, self.trained_db, self.tpl, self.hop, self.norm_opcode, self.mask, self.model_config]], columns=columns)
        if old_exp_id is None: # append (new / ignore `force`)
            configs.to_csv(self.config_file, mode='a', header=False, index=False)
        elif self.force: # delete old and append new
            setup_logger.info('Overwriting the configed experiment.')
            old_config = pd.read_csv(self.config_file)
            new_config = old_config[~old_config.exp_id.isin([old_exp_id])].append(configs)
            new_config.to_csv(self.config_file, index=False)
            shutil.rmtree(f'{self.exp_base}/{old_exp_id}')
        else: raise Exception # debug
        self.__get_train_paths()
        self.model_path = f'{self.exp_base}/{self.exp_id}/models'
        makedirs(self.model_path)
        self.get_models()
        self.last_model = f'{self.model_path}/0'

    def __get_train_paths(self, clear=False):
        self.score_path = f'{self.exp_base}/{self.exp_id}/scores'
        makedirs(self.score_path)
        writer_dir = f'{self.exp_base}/{self.exp_id}/tensorboard/'
        if clear and os.path.exists(writer_dir):
            shutil.rmtree(writer_dir)
        self.writer = SummaryWriter(writer_dir)

    def __get_exp_id(self):
        configs = pd.read_csv(self.config_file)
        exp_id = None
        for _, v in configs.iterrows():
            match = (eval(v.trained_db)==self.trained_db)&(v.tpl==self.tpl)&(v.hop==self.hop)&(v.norm_opcode==self.norm_opcode)&(v['mask']==self.mask)&(eval(v.model_config)==self.model_config)
            if match:
                exp_id = v.exp_id
                break
        if exp_id is not None:
            self.exp_id = exp_id

    def get_models(self, last=False, model_dict=None):
        self.models = glob.glob(f'{self.model_path}/*')
        if last:
            if model_dict is None:
                last_model = glob.glob(f'{self.model_path}/last_epoch_*')
                if self.force and (not last_model):
                    best_model = self.find_best_model()
                    last_model = best_model if best_model else [f'{self.model_path}/0']
                assert len(last_model) == 1, self.model_existence_notice(last=True)
                self.last_model = last_model[0]
            else:
                self.last_model = model_dict
        return self.models

    def set_last_model(self, epoch):
        if os.path.exists(self.last_model):
            os.remove(self.last_model)
        self.last_model = f'{self.model_path}/last_epoch_{epoch+self.start_epoch}'

    def find_best_model(self, keys=['f1']):
        models = self.get_models()
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

    def get_feature_dim(self):
        db = list(self.trained_db)[0]
        flens = get_feature_len(db, self.hop, self.tpl, base_dir=self.graph_base)
        if self.mask == -1:
            return sum(flens)
        if self.mask in [0, 1]:
            return flens[self.mask]
        elif self.mask == 2:
            return 1
        raise ValueError

    def load_model(self, model_path=None):
        num_node_features = self.get_feature_dim()
        global_pool = self.model_config['global_pool']
        dimension = self.model_config['dimension']
        conv_model = GNNStack(num_node_features, dimension, 2, global_pool=global_pool) # hidden_dim=128, num_classes=2
        if model_path is None:
            model_path = self.find_best_model(keys=['f1'])[0]
        conv_model.load_state_dict(torch.load(model_path))
        return conv_model, model_path.split('/')[-1]

    def model_existence_notice(self, last=False):
        if not last:
            return 'Notice that only test mode is supported with this configuration now. Set `continue_train` to `True` if you want to continue to train from the last epoch, or use a new `exp_base` if you want to run a same configured experiment with a differently shuffled data split.'
        else:
            return 'Last training was broken. Set `force` to True if you want to overwrite.'
