# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 19:55:46 2013

@author: edouard
"""

from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.feature_selection import SelectKBest
from addtosklearn import Permutation
from sklearn.svm import SVC
from sklearn.lda import LDA        

# Build sequential Pipeline
pipe = SEQ(SelectKBest(k=2), SVC(kernel="linear"))

# Parallelization 
p = PAR(LDA(),  SVC(kernel="linear"))
p = PAR(*[SelectKBest(k=k) for k in [1, 10, 100]])
p = PAR(*[SVC(kernel=kernel) for kernel in ("linear", "rbf")])
# Combine PAR with sequential Pipeline
p = PAR(LDA(), pipe)
p = PAR(*[SEQ(SelectKBest(k=k), SVC(kernel="linear")) for k in [1, 10, 100]])

[item.name for item in p.children]
# Use Splitters
PAR(KFold, dict(n="y.shape[0]", n_folds=3), LDA())
# Two permutations of 3 folds of univariate filtering of SVM and LDA
import tempfile
import numpy as np
store = tempfile.mktemp()
X = np.asarray([[1, 2], [3, 4], [5, 6], [7, 8], [-1, -2], [-3, -4], [-5, -6], [-7, -8]])
y = np.asarray([1, 1, 1, 1, -1, -1, -1, -1])

methods = [SVC(kernel="linear"), SVC(kernel="rbf"), SVC(kernel="poly")]




_dicts_diff([dict(a=1, b=2, c=3), dict(b=0, c=3)])

# Design of the execution tree
algos = SEQ(SelectKBest(k=2), 
            PAR(LDA(), SVC(kernel="linear")))
algos_cv = PAR(StratifiedKFold, dict(y="y", n_folds=2), algos)
perms = PAR(Permutation, dict(n="y.shape[0]", n_perms=3, apply_on="y"), algos_cv,
           finalize=dict(y=y), store=store)
perms2 = NodeFactory(store=store)

#NFac(fit=f_classif)


# Avoid unnessary compuations
algos = SEQ(PAR(SelectKBest, dict(k=[1, 2, 3]),
                SVC(kernel="linear")))

# run
[(leaf.get_key(2), leaf.top_down(X=X, y=y)) for leaf in perms2]
print leaf.get_key()
print leaf.get_key(2)

r = perms2.bottum_up()
r['SelectKBest/LDA']['y_pred']
np.array(r['SelectKBest/LDA']['y_pred']).shape
