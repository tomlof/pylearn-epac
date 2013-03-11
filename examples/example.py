# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 19:55:46 2013

@author: edouard
"""
# run epac.py

import numpy as np
X = np.asarray([[1, 2], [3, 4], [5, 6], [7, 8], [-1, -2], [-3, -4], [-5, -6], [-7, -8]])
y = np.asarray([1, 1, 1, 1, -1, -1, -1, -1])

from sklearn import datasets
iris = datasets.load_iris()

# Add the noisy data to the informative features
#X = np.hstack((iris.data, np.random.normal(size=(len(iris.data), 20))))
#y = iris.target


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
# The downstream data-flow is a keyword arguments (dict) containing X and y.
# It will pass through each processing node, SelectKBest(k=2) and SVC.
# The Fit:
# Each non-leaf (here SelectKBest  node call the fit method, then apply
# the transformation on the downstream and pass it to the next node. The leaf
# node (here SVC) do not call the transformation.
# The predict:
# Similar sequential tranformation are applied on X, except that the leaf node
# call the predict method.

## Parallelization
## ===============

#from epac import Par
# Multi-classifiers
# -----------------

#    Par    ParMethods (Splitter)
#  /   \
# LDA  SVM  Classifiers (Estimator)
multi = ParMethods(LDA(),  SVC(kernel="linear"))
multi.fit(X=X, y=y)
multi.predict(X=X)

#           Par          ParMethods (Splitter)
#          /  \
# SVM(linear)  SVM(rbf)  Classifiers (Estimator)
svms = ParMethods(*[SVC(kernel=kernel) for kernel in ("linear", "rbf")])
svms.fit(X=X, y=y)
svms.predict(X=X)
svms.bottum_up()

# Parallelize sequential Pipeline: Anova(k best selection) + SVM.
# No collisions between upstream keys, then no aggretation.
#     Par      MultiMethod (Splitter)
#  /   |   \
# 1    5   10  SelectKBest (Estimator)
# |    |    |
# SVM SVM SVM  Classifiers (Estimator)
anovas_svm = ParMethods(*[Seq(SelectKBest(k=k), SVC(kernel="linear")) for k in 
    [1, 5, 10]])
anovas_svm.fit(X=X, y=y)
anovas_svm.predict(X=X)
anovas_svm.bottum_up()
[l.get_key() for l in anovas_svm]
[l.get_key(2) for l in anovas_svm]  # No key 2 collisions, no aggregation


# Parallelize SVM with several parameters.
# Collisions between upstream keys, trig aggretation.

#                    Par                ParGrid (Splitter)
#                  /     \
# SVM(linear, C=1)  .... SVM(rbf, C=10) Classifiers (Estimator)
# ParGrid and PArMethods differ onlys the way they process the upstream
# flow. With ParGrid Children differs only by theire arguments, and thus
# are aggregated toggether
svms = ParGrid(*[SVC(kernel=kernel, C=C) for \
    kernel in ("linear", "rbf") for C in [1, 10]])
svms.fit(X=X, y=y)
svms.predict(X=X, y=y)
svms.bottum_up()
[l.get_key() for l in svms]
[l.get_key(2) for l in svms] # key 2 collisions trig aggregation

# Cross-validation
# ---------------
# CV of LDA
#     CV                  (Splitter)
#  /   |   \
# 0    1    2  Folds      (Slicer)
# |    |    |
# LDA LDA LDA  Classifier (Estimator)
cv_lda = CV(LDA(), n_folds=3, y=y, reducer=SelectAndDoStats())
cv_lda.fit(X=X, y=y)
cv_lda.predict(X=X, y=y)
cv_lda.bottum_up()

# A CV node is a Splitter: it as one child per fold. Each child is a slicer
# ie.: it re-slices the downstream data-flow according into train or test
# sample. When it is called with "fit" it uses the train samples. When it is 
# called with "predict" it uses the test samples.
# If it is called with transform, user has to precise wich sample to use. To
# do that just add a argument sample_set="train" or "test" in the downstream
# data-flow. This argument will be catched by the slicer.
cv_lda.transform(X=X, sample_set="train")
cv_lda.transform(X=X, sample_set="test")

# Permutations + Cross-validation
# -------------------------------
#              Perm                  CV (Splitter)
#         /     |       \
#        0      1       2            Samples (Slicer)
#       |
#     CV                             CV (Splitter)
#  /   |   \
# 0    1    2                        Folds (Slicer)
# |    |    |
# LDA LDA LDA                        Classifier (Estimator)

from epac import Perm, CV, SelectAndDoStats, PvalPermutations, load_node
from epac import *

perms_cv_lda = Perm(CV(LDA(), n_folds=3, reducer=SelectAndDoStats()),
                    n_perms=3, permute="y", y=y, reducer=PvalPermutations())
# Save tree
import tempfile
perms_cv_lda.save(store=tempfile.mktemp())
key = perms_cv_lda.get_key()
# Reload tree
tree = load_node(key)
# Fit
tree.fit(X=X, y=y)
tree.predict(X=X, y=y)
# Save tree with results
tree.save()
# Reload
tree2 = load_node(key)
# Reduce
tree2.bottum_up()
