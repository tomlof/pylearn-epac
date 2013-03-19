# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 13:58:21 2013

@author: edouard.duchesnay@cea.fr
"""

from .workflow import Seq, ParCV, ParPerm, ParMethods, ParGrid, load_workflow
from .workflow import Config  # FIXME
#from .stores import obj_to_dict, dict_to_obj
from .reducers import SummaryStat, PvalPermutations
from . import sklearn_plugins

__all__ = ['Seq',
           'ParCV',
           'ParPerm',
           'ParGrid',
           'ParMethods',
           'Config',
           'load_workflow',
           "SummaryStat",
           "PvalPermutations",
           'sklearn_plugins'
#           "obj_to_dict",
#           "dict_to_obj"
           ]