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
from epac import SummaryStat, PvalPermutations

iris = datasets.load_iris()
# Add the noisy data to the informative features
X = np.hstack((iris.data, np.random.normal(size=(len(iris.data), 20))))
y = iris.target

n_perms = 2
rnd = 0
n_folds = 2
k_values = [1, 5, 10]
# Do it with EPAC
anovas_svm = ParMethods(*[Seq(SelectKBest(k=k), SVC(kernel="linear")) for k in
    k_values])

perms_cv_aov_svm = \
ParPerm(
    ParCV(anovas_svm, n_folds=n_folds,
          reducer=SummaryStat(filter_out_others=False)),
    n_perms=2, permute="y", y=y, random_state=rnd, 
        reducer=PvalPermutations(filter_out_others=False))

# Save tree
import tempfile
perms_cv_aov_svm.save(store=tempfile.mktemp())
key = perms_cv_aov_svm.get_key()
tree = load_workflow(key)
# Fit & Predict
perms_cv_aov_svm.fit_predict(X=X, y=y)
# Save results
perms_cv_aov_svm.save(attr="results")
key = perms_cv_aov_svm.get_key()
# Reload tree, all you need to know is the key
tree = load_workflow(key)
# Reduces results
tree.reduce()

# Do it with sklearn
from sklearn.cross_validation import StratifiedKFold
from epac.sklearn_plugins import Permutation
from sklearn.pipeline import Pipeline

perms = Permutation(n=y.shape[0], n_perms=n_perms, random_state=rnd)

res_lab = ['train_score_y', 'test_score_y', "pred_y", 'true_y']

RES = {"SelectKBest(k=%d)/SVC" % k: {l: [[None]*n_folds]*n_perms 
                for l in res_lab}  for k in k_values}

RES[key]['mean_test_score_y'] = [None]*n_perms
RES[key]['mean_train_score_y'] = [None]*n_perms


perm_nb = 0
for idx in perms:
    y_p = y[idx]
    cv = StratifiedKFold(y=y_p, n_folds=n_folds)
    fold_nb = 0
    for idx_train, idx_test in cv:
        [(idx_train, idx_test) for idx_train, idx_test in cv]
        X_train = X[idx_train, :]
        X_test = X[idx_test, :]
        y_p_train = y_p[idx_train, :]
        y_p_test = y_p[idx_test, :]
        # ANOVA SVM-C
        # 1) anova filter, take 3 best ranked features
        anova_filters = [SelectKBest(k=k) for k in k_values]
        # 2) svm
        clfs = [SVC(kernel='linear') for k in k_values]
        anova_svms = [Pipeline([('anova', anova_filters[i]), ('svm', clfs[i])]) for i in xrange(len(k_values))]
        for i in xrange(len(k_values)):
            k = k_values[i]
            key = "SelectKBest(k=%d)/SVC" % k
            anova_svm = anova_svms[i]
            anova_svm.fit(X_train, y_p_train)
            RES[key]['train_score_y'][perm_nb][fold_nb] = anova_svm.score(X_train, y_p_train)
            RES[key]['test_score_y'][perm_nb][fold_nb] = anova_svm.score(X_train, y_p_test)
            RES[key]['pred_y'][perm_nb][fold_nb] = anova_svm.predict(X)
            RES[key]['true_y'][perm_nb][fold_nb] = y_p_test
        fold_nb += 1
    for key in RES:
        print key, perm_nb, RES[key].keys()
        # Average over folds
        RES[key]['mean_test_score_y'][perm_nb] = \
            np.mean(RES[key]['test_score_y'][perm_nb])
        RES[key]['mean_train_score_y'][perm_nb] = \
            np.mean(RES[key]['train_score_y'][perm_nb])
    perm_nb +=1




RES[key]
