# -*- coding: utf-8 -*-
"""
Created on Thu May 23 15:21:35 2013

@author: ed203246
"""



from sklearn import datasets
from sklearn.svm import LinearSVC as SVM
from sklearn.lda import LDA
from sklearn.feature_selection import SelectKBest
X, y = datasets.make_classification(n_samples=100, n_features=200,
                                    n_informative=2)
X = numpy.random.rand(*X.shape)

from epac import Perms, CV, Methods
perms_cv_svm = Perms(CV(Methods(SVM(loss="l1"), SVM(loss="l2"))), n_perms=100)
perms_cv_svm.run(X=X, y=y)
perms_cv_svm.reduce()

self = perms_cv_svm
key = 'LinearSVC(loss=l1)'
self = PvalPerms()