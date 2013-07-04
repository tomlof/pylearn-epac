# -*- coding: utf-8 -*-
"""
Created on Thu May 23 15:21:35 2013

@author: ed203246
"""

from sklearn import datasets
from sklearn.svm import SVC
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

Xy = dict(X=X, y=y)

from epac import Methods
self = Methods(SVC(C=1), SVC(C=10))
self.run(X=X, y=y)


from epac import Pipe, CV
pipe = Pipe(SelectKBest(k=2), SVC())
cv = CV(pipe,reducer=None)
cv.top_down(X=X, y=y)
cv.reduce()
