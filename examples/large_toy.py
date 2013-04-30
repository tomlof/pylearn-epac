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
n_perms = 500
n_folds = 2
n_folds_nested = 2
random_state = 0
k_values = [2, 3]
C_values = [1, 10]
# ===================
# = With EPAC
# ===================
from epac import ParPerm, ParCV, ParCVGridSearchRefit, Seq, ParGrid
from epac import SummaryStat, PvalPermutations
## CV + Grid search of a pipeline with a nested grid search
pipeline = ParCVGridSearchRefit(*[
              Seq(SelectKBest(k=k),
                  ParGrid(*[SVC(kernel="linear", C=C) for C in C_values]))
              for k in k_values],
              n_folds=n_folds_nested, y=y, random_state=random_state)
wf = ParPerm(
         ParCV(pipeline,
               n_folds=n_folds,
               reducer=SummaryStat(filter_out_others=False)),
         n_perms=n_perms, permute="y", y=y,
         reducer=PvalPermutations(filter_out_others=False),
         random_state=random_state)

wf.fit_predict(X=X, y=y)
wf.reduce()