# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 19:19:48 2013

@author: edouard.duchesnay@cea.fr
"""

import string
import os.path
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
#from sklearn.lda import LDA
from sklearn.feature_selection import SelectKBest
from epac import WF, Seq, ParMethods, ParCV, ParPerm
from epac import SummaryStat, PvalPermutations

iris = datasets.load_iris()
# Add the noisy data to the informative features
X = np.hstack((iris.data, np.random.normal(size=(len(iris.data), 20))))
y = iris.target

n_perms = 2
rnd = 0
n_folds = 2
k_values = [1, 5, 10]

# ===================
# = With EPAC
# ===================
anovas_svm = ParMethods(*[Seq(SelectKBest(k=k), SVC(kernel="linear")) for k in
    k_values])

perms_cv_aov_svm = \
ParPerm(
    ParCV(anovas_svm, n_folds=n_folds,
          reducer=SummaryStat(filter_out_others=False)),
    n_perms=2, permute="y", y=y, random_state=rnd,
    reducer=PvalPermutations(filter_out_others=False))

# Save tree
# ---------
import tempfile
perms_cv_aov_svm.save(store=tempfile.mktemp())
key = perms_cv_aov_svm.get_key()
tree = WF.load(key)
# Fit & Predict
perms_cv_aov_svm.fit_predict(X=X, y=y)
# Save results
perms_cv_aov_svm.save(attr="results")
key = perms_cv_aov_svm.get_key()
# Reload tree, all you need to know is the key
tree = WF.load(key)
# Reduces results
R1 = tree.reduce()

# ===================
# = Without EPAC
# ===================
from sklearn.cross_validation import StratifiedKFold
from epac.sklearn_plugins import Permutation
from sklearn.pipeline import Pipeline

perms = Permutation(n=y.shape[0], n_perms=n_perms, random_state=rnd)

res_lab = ['train_score_y', 'test_score_y', "pred_y", 'true_y']

keys = ["SelectKBest(k=%d)/SVC" % k for k in k_values]

R2 = dict()
for key in keys:
    R2[key] = {l: [[None]*n_folds]*n_perms  for l in res_lab}
    R2[key]['mean_test_score_y'] = [None]*n_perms
    R2[key]['mean_train_score_y'] = [None]*n_perms
    R2[key]['X_train'] = [[None]*n_folds]*n_perms
    R2[key]['X_test'] = [[None]*n_folds]*n_perms
    R2[key]['y_train'] = [[None]*n_folds]*n_perms
    R2[key]['y_test'] = [[None]*n_folds]*n_perms
    R2[key]['idx_train'] = [[None]*n_folds]*n_perms
    R2[key]['idx_test'] = [[None]*n_folds]*n_perms
   
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
            key = keys[i]
            anova_svm = anova_svms[i]
            anova_svm.fit(X_train, y_p_train)
            R2[key]['train_score_y'][perm_nb][fold_nb] = anova_svm.score(X_train, y_p_train)
            R2[key]['test_score_y'][perm_nb][fold_nb] = anova_svm.score(X_train, y_p_test)
            R2[key]['pred_y'][perm_nb][fold_nb] = anova_svm.predict(X)
            R2[key]['true_y'][perm_nb][fold_nb] = y_p_test
            R2[key]['X_train'][perm_nb][fold_nb] = X_train
            R2[key]['X_test'][perm_nb][fold_nb] = X_test
            R2[key]['y_train'][perm_nb][fold_nb] = y_p_train
            R2[key]['y_test'][perm_nb][fold_nb] = y_p_test
            R2[key]['idx_train'][perm_nb][fold_nb] = idx_train
            R2[key]['idx_test'][perm_nb][fold_nb] = idx_test
        fold_nb += 1
    for key in keys:
        # Average over folds
        R2[key]['mean_test_score_y'][perm_nb] = \
            np.mean(R2[key]['test_score_y'][perm_nb])
        R2[key]['mean_train_score_y'][perm_nb] = \
            np.mean(R2[key]['train_score_y'][perm_nb])
    perm_nb +=1

# ===================
# = Comparison
# ===================
rm = os.path.dirname(os.path.dirname(R1.keys()[1]))+"/"
R1 = {string.replace(key, rm, ""):R1[key] for key in R1}

comp = dict()
for key in R1:
    r1 = R1[key]
    r2 = R2[key]
    comp[key] = {k: np.all(np.asarray(r1[k]) == np.asarray(r2[k])) for k in set(r1.keys()).intersection(set(r2.keys()))}

comp


# ===================
# = DEBUG
# ===================
root_to_leaf = tree.get_leftmost_leaf().get_path_from_root()
ds_kwargs = dict(X=X, y=y)
i = 0

self = root_to_leaf[i]
print self, "============================================"
self.fit(X=X, y=y, recursion=False)
ds_kwargs = self.predict(recursion=False, **ds_kwargs)
print np.all(ds_kwargs["X"] == X), ds_kwargs["X"].shape
print np.all(ds_kwargs["y"] == y), ds_kwargs["y"].shape
i += 1

[np.asarray(v) for v in root_to_leaf[i-1].slices.values()]
np.asarray(self.slices['train'])
np.asarray(self.slices['test'])


perm_nb = 0 
fold_nb = 0
R2[key]['X_train'][perm_nb][fold_nb] == ds_kwargs["X"]
R2[key]['X_test'][perm_nb][fold_nb] == ds_kwargs["X"]
R2[key]['y_train'][perm_nb][fold_nb] == ds_kwargs["y"]
R2[key]['y_test'][perm_nb][fold_nb] == ds_kwargs["y"]
R2[key]['idx_train'][perm_nb][fold_nb]
R2[key]['idx_test'][perm_nb][fold_nb]