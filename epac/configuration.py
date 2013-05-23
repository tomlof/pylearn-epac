# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:04:34 2013

@author: edouard.duchesnay@cea.fr
"""

## ================================= ##
## == Configuration class         == ##
## ================================= ##


class conf:
    TRACE_TOPDOWN = False
    STORE_FS_PICKLE_SUFFIX = ".pkl"
    STORE_FS_JSON_SUFFIX = ".json"
    STORE_NODE_PREFIX = "node"
    STORE_EXECUTION_TREE_PREFIX = "execution_tree"
    STORE_STORE_PREFIX = "store"
    KEY_PATH_SEP = "/"
    KEY_PROT_PATH_SEP = "://"  # key storage protocol / path separator
    SUFFIX_JOB = "job"


class debug:
    DEBUG = False
    current = None