#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   learning.py
@Time    :   2020/07/08 14:55:00
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2020
@Desc    :   None
'''

# here put the import lib
import torch
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

def get_device(dev=None):
    if torch.cuda.is_available():
        if dev is None: dev = 0
        device = torch.device(dev)
    else:
        device = torch.device('cpu')
    return device


def pr_curve(labels, scores, title, single=True, figname=None,  legloc='best'):
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    if title:
        plt.title(title)
    
    def single_curve(labels, scores, title):
        precision, recall, _thresholds = precision_recall_curve(labels, scores)
        area = average_precision_score(labels, scores)
        plt.plot(recall, precision, label=title + ': AP={0:0.4f}'.format(area)) 
        
    if single:
        single_curve(labels, scores, title)
    else:
        for i in range(len(labels)):
            single_curve(labels[i], scores[i], title[i])
    
    plt.legend(loc=legloc)
    if figname:
        plt.savefig(figname)
    plt.show()


def metric2scores(TP, FP, TN, FN, f=True):
    correct = TP + TN
    total = correct + FP + FN
    precission = TP / (TP + FP) if (TP + FP)!=0 else 0
    recall = TP / (TP + FN) if (TP + FN)!=0 else 0
    accuracy = correct / total
    if f:
        f1 = fscore(precission, recall, 1)
        f2 = fscore(precission, recall, 2)
        return precission, recall, accuracy, f1, f2
    else:
        return precission, recall, accuracy


def fscore(p, r, beta):
    try:
        return (1+beta*beta)*p*r / ((beta*beta*p)+r)
    except ZeroDivisionError:
        return 0