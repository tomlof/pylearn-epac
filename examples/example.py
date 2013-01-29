# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 19:55:46 2013

@author: edouard
"""
run epac.py

import numpy as np
#X = np.asarray([[1, 2], [3, 4], [5, 6], [7, 8], [-1, -2], [-3, -4], [-5, -6], [-7, -8]])
#y = np.asarray([1, 1, 1, 1, -1, -1, -1, -1])

from sklearn import datasets
iris = datasets.load_iris()

# Add the noisy data to the informative features
X = np.hstack((iris.data, np.random.normal(size=(len(iris.data), 20))))
y = iris.target


from sklearn.svm import SVC
from sklearn.lda import LDA
from sklearn.feature_selection import SelectKBest

## Build sequential Pipeline
## =========================

#from epac import Seq
# Simple sequential pipeline
# 2  SelectKBest
# |
# SVM Classifier
pipe = Seq(SelectKBest(k=2), SVC(kernel="linear"))
pipe.fit(X=X, y=y).predict(X=X)

## Parallelization
## ===============

#from epac import Par
# Multi-classifiers
# -----------------

#    Par    MultiMethod (Splitter)
#  /   \
# LDA  SVM  Classifiers (Estimator)
multi = Par(LDA(),  SVC(kernel="linear"))
multi.fit(X=X, y=y)
multi.predict(X=X)

#           Par          MultiMethod (Splitter)
#          /  \
# SVM(linear)  SVM(rbf)  Classifiers (Estimator)
svms = Par(*[SVC(kernel=kernel) for kernel in ("linear", "rbf")])
svms.fit(X=X, y=y)
svms.predict(X=X)

# Combine Par with sequential Pipeline: Anova(k best selection) + SVM
#     Par      MultiMethod (Splitter)
#  /   |   \
# 1    5   10  SelectKBest (Estimator)
# |    |    |
# SVM SVM SVM  Classifiers (Estimator)
anovas_svm = Par(*[Seq(SelectKBest(k=k), SVC(kernel="linear")) for k in [1, 5, 10]])
anovas_svm.fit(X=X, y=y)
anovas_svm.predict(X=X)

# Cross-validation
# ---------------
# CV of LDA
#     CV                  (Splitter)
#  /   |   \
# 0    1    2  Fold       (Slicer)
# |    |    |
# LDA LDA LDA  Classifier (Estimator)
cv_lda = CV(LDA(), n_folds=3, y=y)
cv_lda.fit(X=X, y=y)
cv_lda.predict(X=X)

self = cv_lda.children[0]
ds_kwargs = dict(X=X, y=y)

# CV of Anova(k best selection) + SVM
cv_anovas_svm = CV(anovas_svm, n_folds=3, y=y)
cv_anovas_svm.fit(X=X, y=y)
cv_anovas_svm.predict(X=X)

# Permutation / Cross-validation of a Pipeline 
# Permutation of CV of Anova(k best selection) + SVM
perms_cv_anovas_svm = Perm(cv_anovas_svm, n_perms=3, y=y)
perms_cv_anovas_svm.fit(X=X, y=y)
perms_cv_anovas_svm.predict(X=X)
#ds_kwargs = dict(X=X)
#self = perms_cv_anovas_svm.children[0]
#self.get_key()
[self for self in perms_cv_anovas_svm]
self.map_outputs


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
