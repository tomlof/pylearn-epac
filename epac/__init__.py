# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 13:58:21 2013

@author: edouard.duchesnay@cea.fr
@author: benoit.da_mota@inria.fr
"""

from epac.workflow.pipeline import Pipe
from epac.workflow.splitters import Perms, Methods, CV, CVBestSearchRefit
from epac.workflow.base import BaseNode, key_pop, key_split
from epac.configuration import conf, debug
from epac.map_reduce.results import ResultSet, Result
from epac.utils import train_test_merge, train_test_split, dict_diff, range_log2
from epac.stores import StoreFs, StoreMem
from epac.map_reduce.mappers import MapperSubtrees
from epac.map_reduce.engine import SomaWorkflowEngine, LocalEngine

from epac.map_reduce.reducers import ClassificationReport, PvalPerms

import sklearn_plugins

__all__ = ['BaseNode',
           'Pipe',
           'Perms',
           'Methods',
           'CV',
           'CVBestSearchRefit',
           'Estimator'
           'ClassificationReport', 'PvalPerms',
           'Result',
           'ResultSet',
           'sklearn_plugins',
           'conf',
           'debug',
           'train_test_split',
           'train_test_merge',
           'key_pop',
           'key_split',
           'dict_diff',
           'StoreFs',
           'StoreMem',
           'range_log2',
           'MapperSubtrees',
           'SomaWorkflowEngine',
           'LocalEngine'
           ]