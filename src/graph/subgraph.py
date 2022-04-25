#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   subgraph.py
@Time    :   2020/07/05 16:29:47
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2020
@Desc    :   Get the subgraph of a specific sensitive api.
'''

# here put the import lib
import networkx as nx
import math
import pandas as pd
import logging
from graph.visual import plot_api_subgraph, plot_highlight_2node


def simplify_tpl(G, center, debug=False, adjust=None, rotate=None):
    if not debug:
        assert adjust is None and rotate is None, 'param `adjust` and `rotate` are only useful for debug'
    types = nx.get_node_attributes(G,'type') # {node: type, ...}
    if debug:
        if type(adjust) == list:a1, a2 = adjust
        else:a1 = a2 = adjust
        print(f'[Input Graph] start simplifing tpl nodes.')
        plot_api_subgraph(G, types, center, a1, rotate)

    edges = list(G.edges())
    for e in edges:
        source, target = e
        if types[source] == 3: # tpl
            G.remove_edge(source, target)
    Gnodes = list(nx.Graph(G).subgraph(c) for c in nx.connected_components(nx.Graph(G)))[0].nodes
    assert center in list(Gnodes)
    Gc = nx.MultiDiGraph(G.subgraph(Gnodes))
    
    if debug:
        print(f'[Pruned Graph] all tpl nodes simplified.')
        types = nx.get_node_attributes(Gc,'type')
        plot_api_subgraph(Gc, types, center, a2, rotate)
        
    return Gc

def check_external(G, ext, center, debug=False, adjust=None):
    if debug:
        print(f'[Input Graph] ext@{ext}, center@{center}.')
        plot_highlight_2node(G, ext, center)
        
    try:
        tmpdel = [e for e in G.in_edges(ext)]
    except Exception:
        print('Error debug', ext in list(G.nodes()))
        
    ext_att = G.nodes(data=True)[ext]
    G.remove_node(ext)
    
    nodes = nx.node_connected_component(nx.Graph(G), center)
    new = nx.MultiDiGraph(G.subgraph(nodes))
    
    new.add_node(ext)
    nx.set_node_attributes(new, {ext: ext_att})
    resume = []
    for i in tmpdel:
        if i[0] in nodes:
            resume.append(i)
    new.add_edges_from(resume)
    
    if debug:
        print(f'[Simplified Graph] ext@{ext} subtree removed.')
        plot_highlight_2node(new, ext, center)
        
    return new

def get_api_name(s):
    first = s.find(' ')
    s = s[first+1:]
    if s.find('@ ') < 0:
        first = s.rfind('>')
        s = s[:first]
    else:
        first = s.find('[access_flags')
        s = s[:first - 1]
    return s

def api_subgraph(node, graph, nodes_type, hop=2, debug=False, apimap=True):
    # 1. n hop neighborhood: `undirected` use both in- and out-neighbors of directed graphs.
    ego_graph = nx.ego_graph(graph, node, radius=hop, undirected=True)
    if debug:
        print(f'{hop} neighborhood of {node}:')
        color = ['r' if n==node else 'b' for n in ego_graph.nodes]
        import matplotlib.pyplot as plt
        nx.draw(ego_graph, node_color=color, with_labels=True); plt.show()
     
    if hop > 2: # 2. prune tpl and external subtrees
        ori_num = len(ego_graph.nodes)
        ego_graph = prune(ego_graph, nodes_type, node, debug=debug)
        logging.info(f'[Prune] node {node}: {len(ego_graph.nodes)} / {ori_num}')
    # else: pass # no need to prune
        
    nodes = ego_graph.nodes
    edges = ego_graph.edges
    
    if apimap: # {node: api, ...}
        api = {}
        for i in list(nodes(data='label')):
            api[i[0]]=get_api_name(i[1])
        
    return [nodes, edges, api] if apimap else [nodes, edges]

def prune(G, nodes_type, center, debug=False, ajust=0):
    subnodes = nodes_type[nodes_type['id'].isin(G.nodes)]
    externals = subnodes[subnodes['type'].isin([0,1])]['id'].tolist()
    externals.remove(center)
    
    if 3 in nodes_type:
        type_att = pd.DataFrame(nodes_type['type']).to_dict('index') # {node: {'type': type}, ...}
        nx.set_node_attributes(G, type_att)
        if debug:
            print(f'Original graph @ {center}:')
            plot_api_subgraph(G, types, center, adjust=ajust, rotate=None)
        G = simplify_tpl(G, center, adjust=ajust)
        if debug:
            print('TPL nodes simplified:')
    elif debug:
        print(f'Original graph @ {center}:')
    
    if debug:
        types = dict(subnodes.set_index(subnodes.id)['type']) # {node: type, ...}
        plot_api_subgraph(G, types, center, adjust=ajust, rotate=None)

    if externals:
        for ext in externals:
            if ext in G.nodes(): # in case the node is deleted
                G = check_external(G, ext, center)
    
    if debug:
        print('External nodes\' subtrees removed:')
        new_types = {}
        for i in G.nodes:
            new_types[i] = types[i]
        plot_api_subgraph(G, new_types, center, adjust=ajust, rotate=None)
    
    return G
    