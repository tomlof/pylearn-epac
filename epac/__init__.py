# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 13:58:21 2013

@author: edouard.duchesnay@cea.fr
@author: benoit.da_mota@inria.fr
"""

from epac.workflow.pipeline import Pipe
from epac.workflow.splitters import CV, Perms, Methods, Grid
from epac.workflow.estimators import CVGridSearchRefit
from epac.workflow.base import xy_split, xy_merge, key_pop
from epac.configuration import conf, debug
from epac.results import Results, Result
from epac.utils import dict_diff, range_log2
from epac.stores import StoreFs, StoreMem

#from epac.workflow import WF, Pipe, CV, Perms, Methods, Grid
#from epac.workflow import CVGridSearchRefit
#from epac.workflow import conf, debug
#from epac.workflow import xy_split, xy_merge
#from epac.utils import dict_diff
#from epac.stores import get_store

#from .stores import obj_to_dict, dict_to_obj
from .reducers import SummaryStat, PvalPerms
from . import sklearn_plugins

__all__ = ['Pipe',
           'CV',
           'Perms',
           'Grid',
           'Methods',
           'CVGridSearchRefit',
           'SummaryStat',
           'PvalPerms',
           'Result',
           'Results',
           'sklearn_plugins',
           'conf',
           'debug',
           'xy_split',
           'key_pop',
           'xy_merge',
           'dict_diff',
           'StoreFs',
           'StoreMem',
           'range_log2'
           ]