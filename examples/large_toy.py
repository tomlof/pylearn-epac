# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 10:06:54 2013

@author: edouard.duchesnay@cea.fr
@author: benoit.da_mota@inria.fr
"""

from sklearn import datasets
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
X, y = datasets.make_classification(n_samples=100, n_features=500,
                                    n_informative=5)


## Realistic example
from epac import ParPerm, ParCV, ParCVGridSearchRefit, Seq, ParGrid
from epac import SummaryStat, PvalPermutations

# CV + Grid search of a pipeline with a nested grid search
pipeline = ParCVGridSearchRefit(*[Seq(SelectKBest(k=k),
                      ParGrid(*[SVC(kernel="linear", C=C)\
                          for C in [.1, 1, 10]]))
                for k in [1, 5, 10]],
           n_folds=5, y=y)

#pipeline.fit_predict(X=X, y=y)

wf = ParPerm(ParCV(pipeline, n_folds=3, reducer=SummaryStat(filter_out_others=False)),
                    n_perms=3, permute="y", y=y, reducer=PvalPermutations(filter_out_others=False))
wf.fit_predict(X=X, y=y)
wf.reduce()
