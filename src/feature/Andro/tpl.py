# coding=utf-8

from feature.LibRadar.literadar import LibRadarLite
from feature.Utils.utils import *

class Tpl():
    def __init__(self, tpl_list, G, outpath, sensitiveapimap, permission, class2init, deepth):
        if type(tpl_list) == list:
            self.tpl_list = tpl_list
        else:
            self.tpl_list = self.get_pkg(tpl_list)
            debug(self.tpl_list)
        self.G = G
        self.sensitiveapimap = sensitiveapimap
        self.TplSensitiveNodeMap = {}
        self.permission = permission
        self.outpath = outpath
        self.class2init = class2init
        self.replacemap = {'android/os/AsyncTask;->execute': (
        'android/os/AsyncTask;->onPreExecute', 'android/os/AsyncTask;->doInBackground'),
                           'android/os/Handler;->sendMessage': ('android/os/Handler;->handleMessage'),
                           'java/lang/Thread;->start': ('java/lang/Runnable;->run')}
        self.deepth = deepth

    def get_pkg(self, apk_path, ratio=0.6):
        lrd = LibRadarLite(apk_path)
        pkgs = []
        try:
            res = lrd.compare()
        except Exception:
            return pkgs
        for i in res:
            p = i['Package']
            try:
                r = eval(i['Match Ratio'])
                if r >= ratio:
                    pkgs.append(p)
                r = '%f' % r
            except KeyError:
                r = 'Undefined'
        return pkgs

    def dfs(self, nodeid):
        async_baseclass = {}
        nodes = list()
        leafs = set()
        nodes.append(nodeid)
        label = get_label(nodeid, self.G)
        node_class = getclass(label)
        # debug(node_class)
        parent_class = getclass(label)
        if node_class in self.class2init:
            async_baseclass.update(self.class2init[node_class])
        dp = 0
        while nodes:
            nodeid = nodes.pop()
            label = get_label(nodeid, self.G)
            for rk in self.replacemap:
                if label.find(rk) >= 0:  # 调用了start函数
                    for ck in self.class2init.keys():
                        if parent_class.find(ck) >= 0:
                            async_baseclass = self.class2init[ck]
                            for asid in async_baseclass.keys():  # 查找对应的run函数
                                funcs = self.replacemap[rk]
                                for func in funcs:
                                    if func.find(async_baseclass[asid]) >= 0:
                                        nodes.append(asid)
                                        debug("get it --->", asid, nodeid)
            leafs.add(nodeid)
            targets = self.G.successors(nodeid)
            if dp < self.deepth:
                for t in targets:
                    if t in leafs:
                        continue
                    nodes.append(t)
            dp = dp + 1
        return leafs

    def getTplSensitiveNode(self, nodeid):
        TplSensitiveNodes = set()
        leafs = self.dfs(nodeid)
        for leaf in leafs:
            if leaf in self.sensitiveapimap:
                TplSensitiveNodes.add(leaf)
        return TplSensitiveNodes

    def writefile(self):
        f = create_csv(self.permission, self.outpath)
        for tplnode in self.TplSensitiveNodeMap:
            sensitiveNodeList = self.TplSensitiveNodeMap[tplnode]
            num_sensitive = len(sensitiveNodeList)
            permap = {}
            for p in self.permission:
                permap[p] = 0
            if num_sensitive != 0:
                for sensitiveNode in sensitiveNodeList:
                    nodepers = self.sensitiveapimap[sensitiveNode]
                    for nodeper in nodepers:
                        permap[nodeper] = permap[nodeper] + 1
                for p in self.permission:
                    permap[p] = permap[p] / num_sensitive
            write_csv(permap, f, tplnode)
        f.close()

    def generate(self):
        self.node_attr = df_from_G(self.G)
        labels = self.node_attr.label
        ids = self.node_attr.id
        i = 0
        self.leafsmap = {}
        while i < len(ids):
            node_id = ids.get(i)
            label = labels.get(i)
            flag = False
            node_class = "L" + getclass(label)
            for tpl in self.tpl_list:
                if node_class.find(tpl) >= 0:
                    flag = True
                    break
            if flag:
                self.TplSensitiveNodeMap[node_id] = self.getTplSensitiveNode(node_id)
            i = i + 1
        self.writefile()
