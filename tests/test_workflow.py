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
from sklearn.lda import LDA
from sklearn.feature_selection import SelectKBest
from epac import WF, Seq, ParMethods, ParCV, ParPerm
from epac import SummaryStat, PvalPermutations
from epac.workflow import conf, debug
conf.DEBUG = True  # set debug to True

iris = datasets.load_iris()
# Add the noisy data to the informative features
#X = np.hstack((iris.data, np.random.normal(size=(len(iris.data), 20))))
#y = iris.target
X = np.asarray([[1, 2, 10, 0], [3, 4, 0, 10], [5, 6, 10, 0], [7, 8, 0, 10], [-1, -2, 10, 0], [-3, -4, 0, 10], [-5, -6, 10, 0], [-7, -8, 0, 10]])
y = np.asarray([1, 1, 1, 1, -1, -1, -1, -1])

n_perms = 2
rnd = 0
n_folds = 2
k_values = [2, 3]#5, 10]

# ===================
# = With EPAC
# ===================
anovas_svm = ParMethods(*[Seq(SelectKBest(k=k), SVC(kernel="linear")) for k in
    k_values])

anovas_svm = ParMethods(*[Seq(SelectKBest(k=k), LDA()) for k in
    k_values])

wf = \
ParPerm(
    ParCV(anovas_svm, n_folds=n_folds,
          reducer=SummaryStat(filter_out_others=False)),
    n_perms=2, permute="y", y=y, random_state=rnd,
    reducer=PvalPermutations(filter_out_others=False))


#wf = ParCV(anovas_svm, n_folds=n_folds, reducer=SummaryStat(filter_out_others=False), y=y)
#wf.children[0].slices
#cv = sklearn.cross_validation.StratifiedKFold(y=y, n_folds=2)

# Save tree
# ---------
# import tempfile
#wf.save(store=tempfile.mktemp())
#key = wf.get_key()
#wf = WF.load(key)
# Fit & Predict
wf.fit_predict(X=X, y=y)
# Save results
#wf.save(attr="results")
#key = wf.get_key()
## Reload tree, all you need to know is the key
#wf = WF.load(key)
## Reduces results
R1 = wf.reduce()

rm = os.path.dirname(os.path.dirname(R1.keys()[1]))+"/"
R1 = {string.replace(key, rm, ""):R1[key] for key in R1}
#WF.load(key).get_key()

# ===================
# = Without EPAC
# ===================
from sklearn.cross_validation import StratifiedKFold
from epac.sklearn_plugins import Permutation
from sklearn.pipeline import Pipeline

keys = R1.keys()
#keys = ["SelectKBest(k=%d)/SVC" % k for k in k_values]

R2 = dict()
for key in keys:
    R2[key] = dict()
    R2[key]['idx_perm'] = [None] * n_perms
    R2[key]['idx_train'] = [[None]*n_folds for i in xrange(n_perms)]
    R2[key]['idx_test'] = [[None]*n_folds for i in xrange(n_perms)]
    R2[key]['X_train'] = [[None]*n_folds for i in xrange(n_perms)]
    R2[key]['X_test'] = [[None]*n_folds for i in xrange(n_perms)]
    R2[key]['y_train'] = [[None]*n_folds for i in xrange(n_perms)]
    R2[key]['y_test'] = [[None]*n_folds for i in xrange(n_perms)]
    R2[key]['pred_y'] = [[None]*n_folds for i in xrange(n_perms)]
    R2[key]['true_y'] = [[None]*n_folds for i in xrange(n_perms)]
    #R2[key] = {l: [[None] * n_folds] * n_perms  for l in res_lab}
    R2[key]['train_score_y'] = [[None]*n_folds for i in xrange(n_perms)]
    R2[key]['test_score_y'] = [[None]*n_folds for i in xrange(n_perms)]
    R2[key]['mean_test_score_y'] = [None] * n_perms
    R2[key]['mean_train_score_y'] = [None] * n_perms


perm_nb = 0
perms = Permutation(n=y.shape[0], n_perms=n_perms, random_state=rnd)
for idx in perms:
    print "perm", perm_nb, "idx", idx
    y_p = y[idx]
    cv = StratifiedKFold(y=y_p, n_folds=n_folds)
    fold_nb = 0
    for idx_train, idx_test in cv:
        print "    cv",fold_nb,"idx Train/text", idx_train, idx_test
        #(idx_train, idx_test) for idx_train, idx_test in cv]
        X_train = X[idx_train, :]
        X_test = X[idx_test, :]
        y_p_train = y_p[idx_train, :]
        y_p_test = y_p[idx_test, :]
        # ANOVA SVM-C
        # 1) anova filter, take 3 best ranked features
        anova_filters = [SelectKBest(k=k) for k in k_values]
        # 2) svm
        #clfs = [SVC(kernel='linear') for k in k_values]
        clfs = [LDA() for k in k_values]
        anova_svms = [Pipeline([('anova', anova_filters[i]), ('svm', clfs[i])]) for i in xrange(len(k_values))]
        for i in xrange(len(k_values)):
            key = keys[i]
            anova_svm = anova_svms[i]
            anova_svm.fit(X_train, y_p_train)
            R2[key]['idx_perm'][perm_nb] = idx
            R2[key]['idx_train'][perm_nb][fold_nb] = idx_train
            R2[key]['idx_test'][perm_nb][fold_nb] = idx_test
            print "    -",key,"perm_nb",perm_nb,"cv",fold_nb,"idx train/test", idx_train, idx_test
            print "    -",R2[key]['idx_train'][perm_nb][fold_nb],R2[key]['idx_test'][perm_nb][fold_nb]
            R2[key]['X_train'][perm_nb][fold_nb] = X_train
            R2[key]['X_test'][perm_nb][fold_nb] = X_test
            R2[key]['y_train'][perm_nb][fold_nb] = y_p_train
            R2[key]['y_test'][perm_nb][fold_nb] = y_p_test
            R2[key]['pred_y'][perm_nb][fold_nb] = anova_svm.predict(X_train)
            R2[key]['true_y'][perm_nb][fold_nb] = y_p_test
            R2[key]['train_score_y'][perm_nb][fold_nb] = anova_svm.score(X_train, y_p_train)
            R2[key]['test_score_y'][perm_nb][fold_nb] = anova_svm.score(X_train, y_p_test)
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


comp = dict()
for key in R1:
    r1 = R1[key]
    r2 = R2[key]
    comp[key] = {k: np.all(np.asarray(r1[k]) == np.asarray(r2[k])) for k in set(r1.keys()).intersection(set(r2.keys()))}

print comp

# ===================
# = DEBUG
# ===================

key = R2.keys()[0]

from epac import ds_split, ds_merge

leaf = wf.get_leftmost_leaf()
print leaf.get_key()
nodes = leaf.get_path_from_root().__iter__()
ds_kwargs = dict(X=X, y=y)

self = nodes.next()
print self, "============================================"
ds_kwargs_train, ds_kwargs_test = ds_split(ds_kwargs)
if hasattr(self, "slices"):
    print self.slices
    print R2[key]['idx_train'][0][0]
    print R2[key]['idx_test'][0][0]
if hasattr(self, "estimator"):
    print "Test equality of input data"
    print np.all(ds_kwargs_train['X'] == R2[key]['X_train'][0][0])
    print np.all(ds_kwargs_test['X'] == R2[key]['X_test'][0][0])
    # Acces to estimator
    self.estimator.fit(**ds_kwargs_train)
    X = ds_kwargs_train["X"]
    y = ds_kwargs_train["y"]
    node = self
    est = self.estimator

if hasattr(self, "estimator") and not self.children:
    
    #from sklearn.utils import atleast2d_or_csr, array2d, check_random_state
    #from sklearn.utils.utils.fixes import unique
    est.fit(X, y)
ds_kwargs = self.fit_predict(recursion=False, **ds_kwargs)

R2[key]['pred_y'][0][0]
R2[key]['X_train'][0][0]

#
#
#
#
#cv = StratifiedKFold(y=y, n_folds=n_folds)
#for tr, te in cv: print tr,te
#
#
#perm_nb = 0 
#fold_nb = 0
#R2[key]['X_train'][perm_nb][fold_nb] == ds_kwargs["X"]
#R2[key]['X_test'][perm_nb][fold_nb] == ds_kwargs["X"]
#R2[key]['y_train'][perm_nb][fold_nb] == ds_kwargs["y"]
#R2[key]['y_test'][perm_nb][fold_nb] == ds_kwargs["y"]
#R2[key]['idx_train'][perm_nb][fold_nb]
#R2[key]['idx_test'][perm_nb][fold_nb]