# -*- coding: utf-8 -*-

#   Copyright 2017 Zachary Marv (马子�?
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


#   DEX Tree
#
#   This script is used to implement the tree node and tree structure.


from ._settings import *
import hashlib
import csv


# tag_rules
labeled_libs = list()
no_lib = list()

with open(FILE_RULE, 'r') as file_rules:
    csv_rules_reader = csv.reader(file_rules, delimiter=',', quotechar='|')
    for row in csv_rules_reader:
        if row[1] == "no":
            no_lib.append(row)
        else:
            labeled_libs.append(row)


class TreeNode(object):
    """
    Tree Node Structure
    {
        sha256  : 02b018f5b94c5fbc773ab425a15b8bbb              // In fact sha256 is the non-hex one
        weight  : 1023                                          // How many APIs in this Node
        pn      : Lcom/facebook/internal                        // Current package name
        parent  : <TreeNode>                                    // Parent node
        children: dict("pn": <TreeNode>)                              // Children nodes
        match   : list( tuple(package_name, match_weight) )     // match lib list
    }
    """
    def __init__(self, n_weight=-1, n_pn="", n_parent=None):
        self.sha256 = ""
        self.weight = n_weight
        self.pn = n_pn
        self.parent = n_parent
        self.children = dict()
        self.match = list()
        self.permissions = set()

    def insert(self, package_name, weight, sha256, permission_list):
        # no matter how deep the package is, add permissions here.
        for permission in permission_list:
            self.permissions.add(permission)
        current_depth = 0 if self.pn == "" else self.pn.count('/') + 1
        target_depth = package_name.count('/') + 1
        if current_depth == target_depth:
            self.sha256 = sha256
            return "F: %s" % package_name
        target_package_name = '/'.join(package_name.split('/')[:current_depth + 1])
        if target_package_name in self.children:
            self.children[target_package_name].weight += weight
            return self.children[target_package_name].insert(package_name, weight, sha256, permission_list)
        else:
            self.children[target_package_name] = TreeNode(n_weight=weight, n_pn=target_package_name, n_parent=self)
            return self.children[target_package_name].insert(package_name, weight, sha256, permission_list)



class Tree(object):
    """
    Tree
    """
    def __init__(self):
        self.root = TreeNode()
        self.db = None
        self.feature = None
        self.feature = dict()
        with open(LITE_DATASET_10, 'r', encoding='UTF-8') as file_rules:
            csv_rules_reader = csv.reader(file_rules, delimiter=',', quotechar='|')
            for row in csv_rules_reader:
                self.feature[row[0]] = row[1:5]

    def insert(self, package_name, weight, sha256, permission_list):
        self.root.insert(package_name, weight, sha256, permission_list)

    def brand(self, package_name, standard_package):
        return self.root.brand(package_name, standard_package)

    def pre_order_res(self, visit, res):
        self._pre_order_res(node=self.root, visit=visit, res=res)

    def _pre_order_res(self, node, visit, res):
        ret = visit(node, res)

        if ret == None or ret < 0:
            return
        else:
            for child_pn in node.children:
                self._pre_order_res(node.children[child_pn], visit, res)

    def pre_order(self, visit):
        self._pre_order(self.root, visit)

    def _pre_order(self, node, visit):
        ret = visit(node)
        if ret < 0:
            return
        else:
            for child_pn in node.children:
                self._pre_order(node.children[child_pn], visit)

    def post_order(self, visit):
        self._post_order(self.root, visit)

    def _post_order(self, node, visit):
        for child_pn in node.children:
            self._post_order(node.children[child_pn], visit)
        visit(node)

    @staticmethod
    def _cal_sha256(node):
        # Ignore Leaf Node
        if len(node.children) == 0 and node.sha256 != "":
            return
        # Everything seems Okay.
        cur_sha256 = hashlib.sha256()
        sha256_list = list()
        for child in node.children:
            sha256_list.append(node.children[child].sha256)
        sha256_list.sort()
        for sha256_item in sha256_list:
            cur_sha256.update(sha256_item.encode())
        node.sha256 = cur_sha256.hexdigest()

        # you could see node.pn here. e.g. Lcom/tencent/mm/sdk/modelpay

    def cal_sha256(self):
        """
        Calculate sha256 for every package
        :return:
        """
        self.post_order(visit=self._cal_sha256)

    def _match(self, node):
        a, c, u = None, None, None

        if node.sha256 in self.feature:
            acu_cur = self.feature[node.sha256]
            a, c, u = acu_cur[3], acu_cur[0], acu_cur[2]

        # if could not find this package in database, search its children.
        if a is None:
            return 1
        # Potential Name is not convincing enough.
        if float(u) < 8 or float(u) / float(c) < 0.3:
            return 2
        flag_not_deeper = False
        for lib in labeled_libs:
            # if the potential package name is the same as full lib path
            # do not search its children
            if lib[0] == a:
                node.match.append([lib, node.weight, int(c)])
                continue
            # If they have the same length but not equal to each other, just continue
            if len(lib[0]) == len(a):
                continue
            # if the potential package name is part of full lib path, search its children
            #   e.g. a is Lcom/google, we could find it as a part of Lcom/google/android/gms, so search its children for
            #       more details
            if len(a) < len(lib[0]) and a == lib[0][:len(a)] and lib[0][len(a)] == '/':
                continue
            # If the lib path is part of potential package name, add some count into parent's match list.
            if len(a) > len(lib[0]) and lib[0] == a[:len(lib[0])] and a[len(lib[0])] == '/':
                depth_diff = a.count('/') - lib[0].count('/')
                cursor = node
                for i in range(depth_diff):
                    # cursor should not be the root, so cursor's parent should not be None.
                    if cursor.parent.parent is not None:
                        cursor = cursor.parent
                    else:
                        # root's parent is None
                        #   This situation exists
                        #   For Example: If it takes Lcom/a/b as Lcom/google/android/gms/ads/mediation/customevent,
                        #   It will find its ancestor until root or None.
                        return 4
                flag = False
                for matc in cursor.match:
                    # if matc[0][0] == lib[0]:
                    if matc[0] == lib:
                        flag = True
                        if matc[1] != cursor.weight:
                            matc[1] += node.weight
                if not flag:
                    cursor.match.append([lib, node.weight, c])
                flag_not_deeper = True
                continue
        if flag_not_deeper:
            return -1
        # Never find a good match, search its children.
        return 5

    def match(self):
        self.pre_order(visit=self._match)

    def _find_untagged(self, node, res):
        # If there's already some matches here, do not search its children. non-sense.
        a, c, u = None, None, None
        if len(node.match) != 0:
            return -1
        if node.sha256 in self.feature:
            acu_cur = self.feature[node.sha256]
            a, c, u = acu_cur[3], acu_cur[0], acu_cur[2]

        if a is None:
            return 1

        # If the package name is already in no_lib list, ignore it and search its children.
        for non_lib in no_lib:
            if non_lib[0] == a:
                return 1
        # Potential Name is not convincing enough. search its children
        if float(u) / float(c) < 0.5 or node.weight < 50 or int(c) < 20:
            return 2

        # JSON support
        utg_lib_obj = dict()            # untagged library object
        utg_lib_obj["Package"] = node.pn
        utg_lib_obj["Standard Package"] = a
        utg_lib_obj["Library"] = "Unknown"
        utg_lib_obj["Popularity"] = int(c)
        utg_lib_obj["Weight"] = node.weight

        res.append(utg_lib_obj)

        # OLD Print
        # print("----")
        # print("Package: %s" % node.pn)
        # print("Match Package: %s" % u)
        # print("Library: Unknown.")
        # print("Popularity: %s" % c)
        # print("API count: %s" % node.weight)

    def find_untagged(self, res):
        self.pre_order_res(visit=self._find_untagged, res=res)

    @staticmethod
    def _get_lib(node, res):
        for matc in node.match:
            if float(matc[1]) / float(node.weight) < 0.1 and matc[0][0] != node.pn:
                continue
            # JSON
            lib_obj = dict()
            lib_obj["Package"] = node.pn  # cpn
            lib_obj["Library"] = matc[0][1] # lib
            lib_obj["Standard Package"] = matc[0][0] # pn
            lib_obj["Type"] = matc[0][2] # tp
            lib_obj["Website"] = matc[0][3] # ch
            lib_obj["Match Ratio"] = "%d/%d" % (matc[1], node.weight) # no similarity in V1
            lib_obj["Popularity"] = matc[2] # dn
            lib_obj["Permission"] = sorted(list(node.permissions))
            res.append(lib_obj)
            # Old Print
            # print("----")
            # print("Package: %s" % node.pn)
            # print("Library: %s" % matc[0][1])
            # print("Standard Package: %s" % matc[0][0])
            # print("Type: %s" % matc[0][2])
            # print("Website: %s" % matc[0][3])
            # print("Similarity: %d/%d" % (matc[1], node.weight))
            # print("Popularity: %d" % matc[2])
            # permission_out = ""
            # for permission in sorted(list(node.permissions)):
            #     permission_out += (permission + ",")
            # if len(permission_out) > 0:
            #     permission_out = permission_out[:-1]
            # print("Permissions:" + permission_out)
        return 0

    def get_lib(self, res):
        self.pre_order_res(visit=self._get_lib, res=res)

