# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:04:34 2013

@author: edouard.duchesnay@cea.fr
"""

## ================================= ##
## == Configuration class         == ##
## ================================= ##
import numpy as np

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
    PREDICTION = "pred"
    SCORE_PRECISION = "score_precision"
    SCORE_RECALL = "score_recall"
    SCORE_RECALL_MEAN = "score_recall_mean"
    SCORE_F1 = "score_f1"
    SCORE_ACCURACY = "score_accuracy"
    BEST_PARAMS = "best_params"
    RESULT_SET = "result_set"
    ML_CLASSIFICATION_MODE = None  # Set to True to force classification mode
    
    @classmethod
    def init_ml(cls, **Xy):
        ## Try to guess if ML tasl is of classification or regression
        if cls.ML_CLASSIFICATION_MODE is None:  # try to guess classif or regression task
            if "y" in Xy:
                y = Xy["y"]
                y_int = y.astype(int)
                if not np.array_equal(y_int, y):
                    cls.ML_CLASSIFICATION_MODE = False
                if np.min(np.bincount(y_int)) < 2:
                    cls.ML_CLASSIFICATION_MODE = False
                cls.ML_CLASSIFICATION_MODE = True
        ## 
class debug:
    DEBUG = False
    current = None