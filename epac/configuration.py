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
    STORE_EXECUTION_TREE_PREFIX = "execution_tree"
    STORE_STORE_PREFIX = "store"
    SEP = "/"
    SUFFIX_JOB = "job"
    KW_SPLIT_TRAIN_TEST = "split_train_test"
    TRAIN = "train"
    TEST = "test"
    TRUE = "true"

class debug:
    DEBUG = False
    current = None