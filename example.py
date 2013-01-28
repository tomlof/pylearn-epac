# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 19:55:46 2013

@author: edouard
"""

X = np.asarray([[1, 2], [3, 4], [5, 6], [7, 8], [-1, -2], [-3, -4], [-5, -6], [-7, -8]])
y = np.asarray([1, 1, 1, 1, -1, -1, -1, -1])

from sklearn.svm import SVC
from sklearn.lda import LDA        
from sklearn.feature_selection import SelectKBest        

# Build sequential Pipeline
pipe = Seq(SelectKBest(k=2), SVC(kernel="linear"))
func = getattr(pipe, "transform")
func(X=X, y=y)
pipe.fit(X, y)
pipe.predict(X)

# Parallelization 
p = Par(LDA(),  SVC(kernel="linear"))
p = Par(*[SelectKBest(k=k) for k in [1, 10, 100]])
p = Par(*[SVC(kernel=kernel) for kernel in ("linear", "rbf")])
# Combine Par with sequential Pipeline
p = Par(LDA(), pipe)

[item.name for item in p.children]

# Spiltters
cv = CV(LDA(), n_folds=3, y=y)
cv = CV(pipe, n_folds=3, y=y)
perm = Perm(pipe, n_perms=3, y=y)
perm = Perm(CV(LDA(), n_folds=3), n_perms=3, y=y)



# Two permutations of 3 folds of univariate filtering of SVM and LDA
import tempfile
import numpy as np
store = tempfile.mktemp()

# Design of the execution tree
algos = Seq(SelectKBest(k=2), 
            Par(LDA(), SVC(kernel="linear")))
algos_cv = Par(StratifiedKFold, dict(y="y", n_folds=2), algos)
perms = Par(Permutation, dict(n="y.shape[0]", n_perms=3, apply_on="y"), algos_cv,
           finalize=dict(y=y), store=store)
perms2 = NodeFactory(store=store)

#NFac(fit=f_classif)


# Avoid unnessary compuations
algos = Seq(Par(SelectKBest, dict(k=[1, 2, 3]),
                SVC(kernel="linear")))

# run
[(leaf.get_key(2), leaf.top_down(X=X, y=y)) for leaf in perms2]
print leaf.get_key()
print leaf.get_key(2)

r = perms2.bottum_up()
r['SelectKBest/LDA']['y_pred']
np.array(r['SelectKBest/LDA']['y_pred']).shape
