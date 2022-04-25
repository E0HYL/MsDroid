#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   file.py
@Time    :   2020/07/07 09:40:38
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2020
@Desc    :   None
'''

# here put the import lib
import os
import errno
import os.path as osp
from pathlib import Path


def makedirs(path):
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e


def find_all_apk(path, end='.apk', layer=None):
    import glob
    if layer is not None:
        all_apk = glob.glob(f"{path}/{'/'.join(['*' for _ in range(layer)])}/*{end}")
    else:
        all_apk = glob.glob(os.path.join(path, '*%s'%end))

        # Get all dirs
        dirnames = [name for name in os.listdir(path)
                if os.path.isdir(os.path.join(path, name))]
        for d in dirnames:
            add_apk = find_all_apk(os.path.join(path, d), end=end)
            all_apk += add_apk
    return all_apk


def search_str(string, filename, firstline=True, reverse=None):
    with open(filename, 'r') as f:
        if firstline:
            first = f.readline()
            pos = first.find(string)
            return pos, first
        else:
            content = f.read()
            if reverse:
                pos = content.rfind(string)
            else:
                pos = content.find(string)
    return pos


def search_pattern(filename, pattern, reverse=True):
    import re

    with open(filename, 'r') as f:
        content = f.readlines()
        if reverse:
            content.reverse()
        for i in content:
            result = re.findall(pattern, i)
            if result:
                break

    return result


def get_project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent
