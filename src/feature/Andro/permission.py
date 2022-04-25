# coding=utf-8
from collections import defaultdict

from . import _settings
from feature.Utils.utils import *

# Permission包含api和contentprovider的部分
# Cursor query(Uri uri, String[] projection, String selection, String[] selectionArgs, String sortOrder)
query_method = "Landroid/content/ContentResolver;->query(Landroid/net/Uri; [Ljava/lang/String; Ljava/lang/String; [Ljava/lang/String; Ljava/lang/String;)Landroid/database/Cursor;"
# Uri insert(Uri uri, ContentValues values)
insert_method = "Landroid/content/ContentResolver;->insert(Landroid/net/Uri; Landroid/content/ContentValues;)Landroid/net/Uri;"
# int delete(Uri uri, String selection, String[] selectionArgs)
delete_method = "Landroid/content/ContentResolver;->delete(Landroid/net/Uri; Ljava/lang/String; [Ljava/lang/String;)I"
# int update(Uri uri, ContentValues values, String selection, String[] selectionArgs)
update_method = "Landroid/content/ContentResolver;->update(Landroid/net/Uri; Landroid/content/ContentValues; Ljava/lang/String; [Ljava/lang/String;)I"

method_dic = {query_method: ['R'], insert_method: ['W'], delete_method: ['W'], update_method: ['W', 'R']}


class Permission():
    def __init__(self, G, path, class_functions, super_dic, implement_dic, dexobj, permission, cppermission, method2nodeMap):
        self.G = G  # call graph
        self.path = path  # csv path to save
        self.class_functions = class_functions
        self.super_dic = super_dic
        self.implement_dic = implement_dic
        self.dexobj = dexobj
        self.sensitiveapimap = {}  # 这个apk中所有的敏感节点与其涉及到的权限
        self.class2runinit = defaultdict(dict)
        self.replacemap = {'Landroid/os/AsyncTask;': ['onPreExecute', 'doInBackground'],
                           'Landroid/os/Handler;': ['handleMessage'], 'Ljava/lang/Runnable;': ['run']}
        self.permission = permission
        self.cp_permission = cppermission
        self.method2nodeMap = method2nodeMap

    def deal_node(self, nodeid):
        permission_node = {}
        targets = self.G.successors(nodeid)
        for t in targets:
            label = get_label(t, self.G)
            for k in method_dic.keys():
                if label.find(k) >= 0:
                    for ki in method_dic[k]:
                        if ki not in permission_node:
                            permission_node[ki] = set()
                        permission_node[ki].add(nodeid)
        return permission_node

    def count_permission(self, name, per_map):
        result = {}
        if name in per_map.keys():
            for p in self.permission:
                result[p] = 0
            pers = per_map[name]
            for per in pers:
                if per not in result.keys():
                    _settings.logger.debug(per + " not in permission list")
                    continue
                result[per] = 1
        return result

    def get_permission(self):
        filename = _settings.cppermissiontxt
        permission = {}
        with open(filename) as f:
            content = f.readline().strip('\n')
            while content:
                cons = content.split(' ')
                if cons[0] not in permission:
                    permission[cons[0]] = set()
                permission[cons[0]].add((cons[1], 'Permission:' + cons[2]))
                content = f.readline().strip('\n')
        return permission

    def get_mappings(self, function, classname):
        mappings = {}
        nodeid, nodelabel = get_nodeid_label(self.G, function)
        if nodeid == "":
            _settings.logger.debug(function)
            _settings.logger.debug(nodelabel)
            return mappings
        # targets = self.G.successors(nodeid)
        classname = 'L' + classname.replace('<analysis.MethodAnalysis ', '') + ';'
        # debug(classname)
        funcList = []
        try:
            tmp = self.class_functions[classname]
            for t in tmp:
                funcList.append(self.method2nodeMap[t][1])
        except KeyError:
            funcList = []
        t = nodelabel
        t_id = nodeid
        external = get_external(nodeid, self.G)
        if external == 1:
            t_class = classname
            if not is_in_funcList(funcList, t):  # t is in super class or system function
                if t_class in self.super_dic.keys():
                    super_class = self.super_dic[t_class]
                    while True:
                        new_label = t.replace(t_class, super_class)
                        try:
                            super_funcList = self.class_functions[super_class]
                            if is_in_funcList(super_funcList, new_label):
                                mappings[t] = (new_label, t_id)
                                break
                            else:
                                t_class = super_class
                                super_class = self.super_dic[t_class]
                        except KeyError:
                            mappings[t] = (new_label, t_id)
                            break
        return mappings


    def substitude(self):  # 替换成子类
        functions = self.node_attr.label
        ids = self.node_attr.id
        for c in self.class_functions: # start method connect run method
            if c in self.implement_dic:
                super_c = self.implement_dic[c]
            else:
                super_c = ""
            # print(super_c)
            if super_c in self.replacemap:
                # print(c, super_c)
                index = 0
                while index < len(ids):
                    func = functions.get(index)
                    if func.find(c + "-><init>(L") >= 0:
                        left = func.find(";-><init>(L") + len(";-><init>(L")
                        right = func.find(";", left)
                        baseclass = func[left: right]
                        # print("baseclass--->", baseclass) # MainActivity
                        # baseclass = getclass(func)
                        index2 = 0
                        func_list = self.replacemap[super_c]
                        while index2 < len(ids):
                            func_tmp = functions.get(index2)
                            for ftmp in func_list:
                                if func_tmp.find(c + "->" + ftmp) >= 0:
                                    self.class2runinit[baseclass].update({ids.get(index2): super_c[1:-1]})
                            index2 = index2 + 1
                        break
                    index = index + 1
        debug("class2init", self.class2runinit)
        mappings = {}
        index = 0
        while index < len(ids):
            label = functions.get(index)
            classname = getclass(label)
            mappings.update(self.get_mappings(label, classname))
            index = index + 1
        # for classname in self.class_functions:
        #     for function in self.class_functions[classname]:
        #         label = self.method2nodeMap[function][1]
        #         mappings.update(self.get_mappings(label, classname))
        # debug("mappings--->", mappings)
        per_map = get_from_csv_gml(_settings.api_file)
        res = {}
        for function in mappings:
            super_function = mappings[function][0]
            for func in per_map:
                if super_function.find(func) >= 0:
                    res[mappings[function][1]] = []
                    for p in per_map[func]:
                        res[mappings[function][1]].append(p)
                    # res[mappings[function][1]] = mappings[function][0]
        debug(res)
        return res  # 所有被替换成子类的敏感API

    def generate(self):
        per_map = get_from_csv_gml(_settings.api_file)
        result_f = create_csv(self.permission, self.path)
        self.node_attr = df_from_G(self.G)
        if type(self.node_attr) == bool and not self.node_attr:
            result_f.close()
            return 2
        getresolver = ";->getContentResolver()Landroid/content/ContentResolver;"
        functions = self.node_attr.label
        ids = self.node_attr.id

        substitude_permission = self.substitude()  # 子类涉及敏感API
        # 获取contentprovider相关的permission
        node_cp_permission = defaultdict(list)
        java_class = {}  # need to generate java file
        for i in range(len(ids)):
            function = functions.get(i)
            # debug(function)
            if function.find(getresolver) >= 0:
                node_id = ids.get(i)
                nodes = n_neighbor(node_id, self.G)
                debug(function, nodes)
                for node in nodes:
                    node_permission = self.deal_node(node)
                    if node_permission:
                        label = get_label(node, self.G)
                        left = label.find(' ')
                        right = label.find('->')
                        function_class = label[left + 1: right]
                        debug(function_class, node_permission)
                        java_class.update({function_class: node_permission})
        debug("java_class", java_class)
        for method in self.dexobj.get_methods():
            if str(method.get_class_name()) in java_class:
                current_class = self.dexobj.get_class(method.get_class_name())
                content = str(current_class.get_source())
                try:
                    node_permission = java_class.pop(method.get_class_name())
                except Exception:
                    _settings.logger.error("%s has error method name %s"%(self.path, method.get_class_name()))
                    continue
                if content.find('content://') >= 0:
                    for per in self.cp_permission.keys():
                        if content.find(per) >= 0:
                            pers = self.cp_permission[per]
                            for p in pers:
                                if p[0] in node_permission:
                                    for n_id in node_permission[p[0]]:
                                        node_cp_permission[n_id].append(p[1])
        debug("node_cp_permission", node_cp_permission)
        i = 0
        while i < len(ids):
            s = functions.get(i)
            s = node2function(s)
            p = self.count_permission(s, per_map)
            node_id = ids.get(i)
            if node_id in node_cp_permission:  # 与content provider相关的权限
                for per in self.permission:
                    p[per] = 0
                for per in node_cp_permission[node_id]:
                    p[per] = 1
            if node_id in substitude_permission:  # 子类是敏感的API
                for per in self.permission:
                    p[per] = 0
                for per in substitude_permission[node_id]:
                    p[per] = 1
            if p != {}:
                write_csv(p, result_f, node_id)
                node_permission = []
                for k in p:
                    if p[k] == 1:
                        node_permission.append(k)
                self.sensitiveapimap.update({node_id: node_permission})
            i += 1
        result_f.close()
        return 0

    def getsensitive_api(self):  # 获取这个apk所有敏感节点的api列表
        return self.sensitiveapimap

    def getPermissionList(self):
        return self.permission

    def getClass2init(self):
        return self.class2runinit
