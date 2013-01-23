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
PAR(LDA(),  SVC(kernel="linear"))
PAR(KFold, dict(n="y.shape[0]", n_folds=3), LDA())
# Two permutations of 3 folds of univariate filtering of SVM and LDA
import tempfile
import numpy as np
store = tempfile.mktemp()
X = np.asarray([[1, 2], [3, 4], [5, 6], [7, 8], [-1, -2], [-3, -4], [-5, -6], [-7, -8]])
y = np.asarray([1, 1, 1, 1, -1, -1, -1, -1])
methods = [SVC(kernel="linear"), SVC(kernel="rbf"), SVC(kernel="poly")]


def _dict_check_same_keys(dicts):
    k = set(dicts[0].keys())
    for d in dicts[1:]:
        diff = k ^ set(d.keys())
        if len(diff):
            return False
    return True

_dict_check_same_keys([m.__dict__ for m in methods])


(k ^ set(d.keys()) for d)

'poly'
a.__dict__
b.__dict__

def _list_diff(l1, l2):
    return [item for item in l1 if not item in l2]


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
