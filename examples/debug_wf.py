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
C_values = [1, 2]

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
wf = ParCV(pipeline,
               n_folds=n_folds,
               reducer=SummaryStat(filter_out_others=True))

from epac import StoreFs

store = StoreFs(dirpath=tempfile.mkdtemp(), clear=True)
wf.save(store=store)
wf.fit_predict(X=X, y=y)
r1 = wf.reduce().values()[0]

wf_loaded = store.load()
wf_loaded.fit_predict(X=X, y=y)
r2 = wf_loaded.reduce().values()[0]

np.all([np.all(np.asarray(r1[k])==np.asarray(r2[k])) for k in r1])