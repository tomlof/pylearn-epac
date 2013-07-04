# -*- coding: utf-8 -*-
"""
Created on Thu May 23 15:21:35 2013

@author: ed203246
"""

from sklearn import datasets
from sklearn.svm import LinearSVC as SVM
from sklearn.lda import LDA
from sklearn.feature_selection import SelectKBest
X, y = datasets.make_classification(n_samples=12, n_features=10,
                                    n_informative=2)

# Build sequential Pipeline
# -------------------------
# 2  SelectKBest (Estimator)
# |
# SVM Classifier (Estimator)

#run -i epac/workflow/base.py
#run -i epac/workflow/estimators.py
#run -i epac/utils.py
#run -i epac/map_reduce/results.py
#run -i epac/map_reduce/reducers.py


import numpy as np
from epac.workflow.base import BaseNode, key_push, key_split
from epac.utils import _func_get_args_names, train_test_merge, train_test_split, _dict_suffix_keys
from epac.utils import _sub_dict, _as_dict
from epac.map_reduce.results import ResultSet, Result
from epac.stores import StoreMem
from epac.configuration import conf
from epac.map_reduce.reducers import ClassificationReport


Xy = dict(X=X, y=y)

from epac import Pipe, CVBestSearchRefit, Methods
# CV + Grid search of a simple classifier
self = CVBestSearchRefit(Methods(*[SVM(C=C) for C in [1, 10]]))
self.run(X=X, y=y)
self.reduce()


from epac import Pipe, CV
pipe = Pipe(SelectKBest(k=2), SVC())
cv = CV(pipe,reducer=None)
cv.top_down(X=X, y=y)
cv.reduce()
