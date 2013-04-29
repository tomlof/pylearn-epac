# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 10:06:54 2013

@author: edouard.duchesnay@cea.fr
@author: benoit.da_mota@inria.fr
"""

from sklearn import datasets
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
X, y = datasets.make_classification(n_samples=100, n_features=500,
                                    n_informative=5)

# ===================
# = With EPAC
# ===================

## Realistic example
from epac import ParPerm, ParCV, ParCVGridSearchRefit, Seq, ParGrid
from epac import SummaryStat, PvalPermutations

## CV + Grid search of a pipeline with a nested grid search
pipeline = ParCVGridSearchRefit(*[
              Seq(SelectKBest(k=k),
                  ParGrid(*[SVC(kernel="linear", C=C) for C in [.1, 1, 10]]))
              for k in [1, 5, 10]],
              n_folds=5, y=y)

#pipeline.fit_predict(X=X, y=y)

wf = ParPerm(
         ParCV(
             pipeline, n_folds=3, reducer=SummaryStat(filter_out_others=False)),
         n_perms=3, permute="y", y=y, reducer=PvalPermutations(filter_out_others=False))
wf.fit_predict(X=X, y=y)
wf.reduce()

# ===================
# = Without EPAC
# ===================
from sklearn.cross_validation import StratifiedKFold
from epac.sklearn_plugins import Permutation
from sklearn.pipeline import Pipeline
keys = R1.keys()
import re

def key_to_k(key):
    find_k = re.compile(u'k=([0-9])')
    return int(find_k.findall(key)[0])
# ANOVA SVM-C
clfs = {key: Pipeline([('anova', SelectKBest(k=key_to_k(key))),
                       ('svm', SVC(kernel="linear"))]) for key in keys}
R2 = dict()
for key in keys:
    R2[key] = dict()
    R2[key]['pred_y'] = [[None] * n_folds for i in xrange(n_perms)]
    R2[key]['true_y'] = [[None] * n_folds for i in xrange(n_perms)]
    R2[key]['train_score_y'] = [[None] * n_folds for i in xrange(n_perms)]
    R2[key]['test_score_y'] = [[None] * n_folds for i in xrange(n_perms)]
    R2[key]['mean_test_score_y'] = [None] * n_perms
    R2[key]['mean_train_score_y'] = [None] * n_perms
perm_nb = 0
perms = Permutation(n=y.shape[0], n_perms=n_perms, random_state=rnd)
for idx in perms:
    y_p = y[idx]
    cv = StratifiedKFold(y=y_p, n_folds=n_folds)
    fold_nb = 0
    for idx_train, idx_test in cv:
        X_train = X[idx_train, :]
        X_test = X[idx_test, :]
        y_p_train = y_p[idx_train, :]
        y_p_test = y_p[idx_test, :]
        for key in keys:
            clf = clfs[key]
            clf.fit(X_train, y_p_train)
            R2[key]['pred_y'][perm_nb][fold_nb] = clf.predict(X_test)
            R2[key]['true_y'][perm_nb][fold_nb] = y_p_test
            R2[key]['train_score_y'][perm_nb][fold_nb] =\
                clf.score(X_train, y_p_train)
            R2[key]['test_score_y'][perm_nb][fold_nb] =\
                clf.score(X_test, y_p_test)
        fold_nb += 1
    for key in keys:
        # Average over folds
        R2[key]['mean_test_score_y'][perm_nb] = \
            np.mean(np.asarray(R2[key]['test_score_y'][perm_nb]), axis=0)
        R2[key]['mean_train_score_y'][perm_nb] = \
            np.mean(np.asarray(R2[key]['train_score_y'][perm_nb]), axis=0)
            #np.mean(R2[key]['train_score_y'][perm_nb])
    perm_nb += 1