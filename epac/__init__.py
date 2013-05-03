# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 13:58:21 2013

@author: edouard.duchesnay@cea.fr
@author: benoit.da_mota@inria.fr
"""

from epac.workflow.base import WF
from epac.workflow.pipeline import Seq
from epac.workflow.splitters import ParCV, ParPerm, ParMethods, ParGrid
from epac.workflow.estimators import ParCVGridSearchRefit
from epac.workflow.base import conf, debug
from epac.workflow.base import xy_split, xy_merge
from epac.utils import dict_diff, range_log2
from epac.stores import get_store

#from epac.workflow import WF, Seq, ParCV, ParPerm, ParMethods, ParGrid
#from epac.workflow import ParCVGridSearchRefit
#from epac.workflow import conf, debug
#from epac.workflow import xy_split, xy_merge
#from epac.utils import dict_diff
#from epac.stores import get_store

#from .stores import obj_to_dict, dict_to_obj
from .reducers import SummaryStat, PvalPermutations
from . import sklearn_plugins

__all__ = ['WF',
           'Seq',
           'ParCV',
           'ParPerm',
           'ParGrid',
           'ParMethods',
           'ParCVGridSearchRefit',
           'SummaryStat',
           'PvalPermutations',
           'sklearn_plugins',
           'conf',
           'debug',
           'xy_split',
           'xy_merge',
           'dict_diff',
           'get_store',
           'range_log2'
           ]