# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 19:19:48 2013

@author: edouard.duchesnay@cea.fr
"""

import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.lda import LDA

from sklearn.feature_selection import SelectKBest

from epac import Seq, ParMethods, ParCV, ParPerm
from epac import load_workflow
from epac import SelectAndDoStats, PvalPermutations

iris = datasets.load_iris()
# Add the noisy data to the informative features
X = np.hstack((iris.data, np.random.normal(size=(len(iris.data), 20))))
y = iris.target

anovas_svm = ParMethods(*[Seq(SelectKBest(k=k), SVC(kernel="linear")) for k in 
    [1, 5, 10]])

perms_cv_aov_svm = ParPerm(ParCV(anovas_svm, n_folds=2, reducer=SelectAndDoStats()),
                    n_perms=2, permute="y", y=y, reducer=PvalPermutations())

# Save tree
import tempfile
perms_cv_aov_svm.save(store=tempfile.mktemp())
key = perms_cv_aov_svm.get_key()
tree = load_node(key)
# Fit & Predict
perms_cv_aov_svm.fit(X=X, y=y)
perms_cv_aov_svm.predict(X=X, y=y)
# Save results
perms_cv_aov_svm.save(attr="results")
key = perms_cv_aov_svm.get_key()
# Reload tree, all you need to know is the key
tree = load_node(key)
# Reduces results
tree.bottum_up()
