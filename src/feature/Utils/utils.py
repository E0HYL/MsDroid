# coding=utf-8
import csv
import os

from feature.Andro._settings import verbose


def debug(*args):
    if verbose:
        print(args)


def create_csv(smali_opcode, path):  # 创建csv
    f = open(path, 'w+', newline='')
    csv_write = csv.writer(f)
    csv_head = ['id'] + smali_opcode
    csv_write.writerow(csv_head)
    return f


def write_csv(opcode, f, id):  # 写csv
    csv_write = csv.writer(f)
    data_row = [id]
    for op in opcode.keys():
        data_row.append(opcode[op])
    csv_write.writerow(data_row)


def df_from_G(G):
    import pandas as pd
    df = pd.DataFrame(G.nodes(data=True))
    try:
        attr = df[1].apply(pd.Series)
    except KeyError:
        return False
    node_attr = pd.concat([df[0], attr], axis=1)
    node_attr = node_attr.rename(columns={0: 'id'})
    return node_attr


def read_permission(path):  # 获取所有的permission
    permission = []
    with open(path) as f:
        line = f.readline()
        while line:
            line = line.strip('\n')
            permission.append(line)
            line = f.readline()
    return permission


def n_neighbor(node, graph, hop=1):  # 节点的一跳邻�?    
    import networkx as nx
    ego_graph = nx.ego_graph(graph, node, radius=hop, undirected=True)
    nodes = ego_graph.nodes
    return nodes


def get_label(node_id, G):
    return G.nodes[node_id]['label']


def get_from_csv_gml(filename):  # 从API.csv获取每个API对应的权限mapping
    per_value = {}
    with open(filename, "r") as csvFile:
        reader = csv.reader(csvFile)
        for item in reader:
            if reader.line_num == 1:
                continue
            name = item[0]
            per = item[1]
            # value[name] = functionname
            if name not in per_value.keys():
                per_value[name] = [per]
            else:
                per_value[name].append(per)
    return per_value


def node2function(s):  # 根据节点名称获取函数名称
    first = s.find(' ')
    s = s[first + 1:]
    if s.find('@ ') < 0:
        first = s.rfind('>')
        s = s[:first]
    else:
        first = s.find('[access_flags')
        s = s[:first - 1]
    return s


def getclass(functionname):
    index = functionname.find(';->')
    return functionname[len('<analysis.MethodAnalysis L'):index]


def getfunction(filename):  # 获取类名和函数名
    with open(filename) as f:
        line = f.readline().strip('\n')
        line = line.replace('# ', '')
        right = line.find(';->')
        classname = line[:right]
        right = line.find('[access_flags')
        if right < 0:
            function = line
        else:
            function = line[:right - 1]
        return classname, function


def get_nodeid_label(G, function):  # 函数名获取id和label
    if type(function) == int:
        return function, G.nodes[function]['label']
    nodes = G.nodes
    for node in nodes:
        label = G.nodes[node]['label']
        if label.find(function) >= 0:
            return node, label
    return "", ""


def is_in_funcList(funcList, t):  # 节点是否再函数列表中
    for f in funcList:
        if t.find(f) >= 0:
            return True
    return False


def get_label(node_id, G):
    return G.nodes[node_id]['label']


def get_external(nodeid, G):
    return G.nodes[nodeid]['external']


def get_codesize(nodeid, G):
    return G.nodes[nodeid]['codesize']


def find_all_apk(path, end='.apk', layer=None):
    import glob
    if layer is not None:
        all_apk = glob.glob(f"{path}/{'/'.join(['*' for _ in range(layer)])}/*{end}")
    else:
        all_apk = glob.glob(os.path.join(path, '*%s' % end))

        # Get all dirs
        dirnames = [name for name in os.listdir(path)
                    if os.path.isdir(os.path.join(path, name))]
        for d in dirnames:
            add_apk = find_all_apk(os.path.join(path, d), end=end)
            all_apk += add_apk
    return all_apk
