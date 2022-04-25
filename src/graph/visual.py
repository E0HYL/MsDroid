#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   visual.py
@Time    :   2020/07/05 16:30:30
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2020
@Desc    :   Visualization utils for api subgraph.
'''

# here put the import lib
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def group_pos(pos, typenodes, rotate=None):
    # prep center points (along circle perimeter) for the clusters
    rotate = np.pi/128 if rotate is None else rotate
    angs = np.linspace(0+rotate, 2*np.pi+rotate, 1+len(typenodes))
    repos = []
    rad = 3     # radius of circle
    for ea in angs:
        if ea  > 0:
            #print(rad*np.cos(ea), rad*np.sin(ea))  # location of each cluster
            repos.append(np.array([rad*np.cos(ea), rad*np.sin(ea)]))

    for ea in pos.keys():
        for t in typenodes:
            if ea in typenodes[t]:
                r = t
                break
        pos[ea] += repos[r]
    
    return pos

def group_cirpos(pos, typenodes):
    radii = [60, 60, 15, 45]  # for concentric circles

    for ea in pos.keys():
        new_r = 1
        for t in typenodes:
            if ea in typenodes[t]:
                new_r = radii[t]
                break
        pos[ea] *= new_r   # reposition nodes as concentric circles
    
    return pos


def plot_highlight_2node(G, ext, center):
    color = ['pink' if n==ext or n==center else 'lightblue' for n in G.nodes]
    nx.draw(G, with_labels=True, node_color=color)
    plt.show()

       
def plot_api_subgraph(G, types, center, adjust=1, rotate=None, **kwargs):
    '''
        `G`: sensitive api centered subgraph
        `types`: dict object indicates the type of each node {node: type, ...}
        `center`: sensitive api of the subgraph
        `adjust`: choices for layout position; `rotate` is only useful when `adjust` = 1
    '''
    color_map = [(1, 0, 0, 0.08), 'pink', 'cornflowerblue', 'lightskyblue']
    typenodes = {i:[] for i in range(1+max(set(types.values())))} # {type:[node, ...], ...}
    for i in types:
        typenodes[types[i]].append(i)

    shape = ['o','o','s','p']
    if adjust == 0:
        pos = nx.circular_layout(G)
    elif adjust == 1:
        pos = group_pos(nx.circular_layout(G), typenodes, rotate=rotate)
    elif adjust == 2:
        pos = group_cirpos(nx.circular_layout(G), typenodes)
    else:
        pos = nx.spring_layout(G)    

    _, ax = plt.subplots(figsize=(8, 8))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    for i in typenodes:
        nx.draw_networkx_nodes(G, pos=pos, with_labels=True, nodelist=typenodes[i]#, node_size=500
                               , node_shape=shape[i], node_color=[color_map[i] for j in range(len(typenodes[i]))])
    nx.draw_networkx_nodes(G, pos=pos, with_labels=True, nodelist=[center], node_color='r')
    nx.draw_networkx_labels(G, pos=pos)
    if kwargs.get('edge_cmap') is None:
        nx.draw_networkx_edges(G, pos=pos)
    else:  
        # ax = plt.gca()
        D = kwargs.get('edge_color')
        RdPu = plt.get_cmap('GnBu') # https://predictablynoisy.com/matplotlib/gallery/color/colormap_reference.html
        norm = plt.Normalize(D.min(), D.max())
        for source, target, data in G.edges(data=True):
            ax.annotate(
                '', xy=pos[target], xycoords='data', xytext=pos[source],
                textcoords='data', 
                color=data['att'],
                arrowprops=dict(
                    arrowstyle="->",
                    alpha=max(data['att'],0.1),
                    color=RdPu(norm(data['att'])),
                    # connectionstyle="arc3,rad=0.1",
                ))
    if kwargs.get('labels') is not None:
        nx.draw_networkx_labels(G, pos=pos, labels=kwargs.get('labels'), 
        font_size=7, alpha=0.7)
    plt.show()
