# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 13:58:21 2013

@author: ed203246
"""

"""
The :mod:`sklearn.feature_selection` module implements feature selection
algorithms. It currently includes univariate filter selection methods and the
recursive feature elimination algorithm.
"""

from .epac import SEQ
from .epac import PAR
from .epac import NodeFactory
from .addtosklearn import Permutation


__all__ = ['SEQ',
           'PAR',
           'RFECV',
           'NodeFactory',
           'Permutation']
