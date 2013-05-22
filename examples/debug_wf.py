# -*- coding: utf-8 -*-
"""
Created on Wed May 22 12:27:30 2013

@author: ed203246
"""

import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.lda import LDA
from sklearn.feature_selection import SelectKBest
#X, y = datasets.make_classification(n_samples=12, n_features=10,
#                                    n_informative=2)
#np.savez("/tmp/Xy_n12-p10.npz", X=X, y=y)
#np.savez("/tmp/Xy_n12-p10.npz", X=X, y=y)
Xy = np.load("/tmp/Xy_n12-p10.npz")
X = Xy["X"]
y = Xy["y"]


n_folds = 2
n_perms = 2
n_folds_nested = 5
k_values = [1, 2]# 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
C_values = [1, 10]

## 2) Build Workflow
    ## =================
from epac import ParPerm, ParCV, ParCVGridSearchRefit, Seq, ParGrid
from epac import SummaryStat, PvalPermutations
## CV + Grid search of a pipeline with a nested grid search
pipeline = ParCVGridSearchRefit(*[
              Seq(SelectKBest(k=k),
                  ParGrid(*[SVC(kernel="linear", C=C) for C in C_values]))
              for k in k_values],
              n_folds=n_folds_nested)

#print pipeline.stats(group_by="class")
self = ParPerm(
         ParCV(pipeline,
               n_folds=n_folds,
               reducer=SummaryStat(filter_out_others=True)),
         n_perms=n_perms, permute="y",
         reducer=PvalPermutations(filter_out_others=True))
         
#self.fit_predict(X=X, y=y)


#from epac import StoreMem
#from epac import debug, conf
#conf.DEBUG = True
#conf.TRACE_TOPDOWN = True
self.fit_predict(X=X, y=y)
#root = self
#self = debug.current
#Xy = debug.Xy

self = debug.current.children[0]
self.children[1].reduce()
self.children[1].fit_predict(**Xy)

from epac import StoreFs
store = StoreFs("/tmp/toto", clear=True)
self.save(store=store)
self.fit_predict(X=X, y=y)
tree = store.load()
tree.reduce()