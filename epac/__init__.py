# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 13:58:21 2013

@author: edouard.duchesnay@cea.fr
"""


from .workflow import WF, Seq, ParCV, ParPerm, ParMethods, ParGrid
from .workflow import CVGridSearchRefit
from .workflow import conf, debug
from .workflow import xy_split, xy_merge
from .utils import dict_diff

#from .stores import obj_to_dict, dict_to_obj
from .reducers import SummaryStat, PvalPermutations
from . import sklearn_plugins

__all__ = ['WF',
           'Seq',
           'ParCV',
           'ParPerm',
           'ParGrid',
           'ParMethods',
           'CVGridSearchRefit',
           'SummaryStat',
           'PvalPermutations',
           'sklearn_plugins',
           'conf',
           'debug',
           'xy_split',
           'xy_merge',
           'dict_diff'
           ]