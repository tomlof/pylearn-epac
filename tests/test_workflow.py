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
from epac.workflow import conf, debug, ds_split
conf.DEBUG = True  # set debug to True

iris = datasets.load_iris()
# Add the noisy data to the informative features
#X = np.hstack((iris.data, np.random.normal(size=(len(iris.data), 20))))
#y = iris.target
#X = np.asarray([[1, 2, 10, 0], [3, 4, 0, 10], [5, 6, 10, 0], [7, 8, 0, 10], [-1, -2, 10, 0], [-3, -4, 0, 10], [-5, -6, 10, 0], [-7, -8, 0, 10]])
#y = np.asarray([1, 1, 1, 1, -1, -1, -1, -1])

y = np.asarray([1, 0, 0, 0, 0, 1, 1, 1, 0, 1])
X = np.asarray([[-0.72410991, -0.16117523, -2.71482769,  1.5102451 , -2.41200339],
       [-0.28835824, -1.36765364,  1.10712306, -1.68043299,  0.46955062],
       [-0.04176457,  0.12350139,  0.31700643, -0.40674412,  0.17038624],
       [-0.16945109,  1.55585327,  0.34532922, -0.66917005,  0.07643056],
       [-0.63092503,  1.58108263, -0.48606433, -0.6439051 , -0.87337035],
       [ 0.2967244 ,  1.89077497, -0.10248979,  0.64807938,  0.19437294],
       [ 1.25326113, -0.85359701,  1.46948375,  0.75351193,  2.06421049],
       [-0.79653103,  0.64577517, -3.77550405,  2.48420756, -3.16897176],
       [-0.36567546,  0.94230265,  0.35040889, -1.03236683, -0.09308279],
       [ 0.77428074, -0.41138325,  0.40265534,  0.99235342,  0.94512705]])

#X, y = datasets.make_classification(n_samples=10, n_features=5, n_informative=2)
n_perms = 3
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
    n_perms=n_perms, permute="y", y=y, random_state=rnd,
    reducer=PvalPermutations(filter_out_others=False))


wf = \
ParPerm(
    ParCV(LDA(), n_folds=n_folds,
          reducer=SummaryStat(filter_out_others=False)),
    n_perms=n_perms, permute="y", y=y, random_state=rnd,
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
wf.fit_predict(X=X, y=y)  # re-run
# Save results
#wf.save(attr="results")
#key = wf.get_key()
## Reload tree, all you need to know is the key
#wf = WF.load(key)
## Reduces results
conf.DEBUG = conf.VERBOSE = True
R1 = wf.reduce()

self = wf.get_node('ParPerm/Perm(nb=0)/ParCV')
#ICI
#leaf = wf.get_leftmost_leaf()

rm = os.path.dirname(os.path.dirname(R1.keys()[0]))+"/"
R1 = {string.replace(key, rm, ""):R1[key] for key in R1}
keys = R1.keys()

#WF.load(key).get_key()

# ===================
# = Without EPAC
# ===================
from sklearn.cross_validation import StratifiedKFold
from epac.sklearn_plugins import Permutation
from sklearn.pipeline import Pipeline

keys = R1.keys()

# ANOVA SVM-C
# 1) anova filter, take 3 best ranked features
#anova_filters = [SelectKBest(k=k) for k in k_values]
# 2) svm
#clfs = [SVC(kernel='linear') for k in k_values]
clfs = {key: LDA() for key in keys}
#anova_svms = [Pipeline([('anova', anova_filters[i]), ('svm', clfs[i])]) for i in xrange(len(k_values))]
#anova_svms = clfs
    
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
for idx in perms: print idx

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
        for key in keys:
            clf = clfs[key]
            clf.fit(X_train, y_p_train)
            R2[key]['idx_perm'][perm_nb] = idx
            R2[key]['idx_train'][perm_nb][fold_nb] = idx_train
            R2[key]['idx_test'][perm_nb][fold_nb] = idx_test
            print "    -",key,"perm_nb",perm_nb,"cv",fold_nb,"idx train/test", idx_train, idx_test
            print "    -",R2[key]['idx_train'][perm_nb][fold_nb],R2[key]['idx_test'][perm_nb][fold_nb]
            R2[key]['X_train'][perm_nb][fold_nb] = X_train
            R2[key]['X_test'][perm_nb][fold_nb] = X_test
            R2[key]['y_train'][perm_nb][fold_nb] = y_p_train
            R2[key]['y_test'][perm_nb][fold_nb] = y_p_test
            R2[key]['pred_y'][perm_nb][fold_nb] = clf.predict(X_test)
            R2[key]['true_y'][perm_nb][fold_nb] = y_p_test
            R2[key]['train_score_y'][perm_nb][fold_nb] = clf.score(X_train, y_p_train)
            R2[key]['test_score_y'][perm_nb][fold_nb] = clf.score(X_test, y_p_test)
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
k = 'pred_y'
np.all(np.asarray(r1[k]) == np.asarray(r2[k]))

# ===================
# = DEBUG
# ===================

key = R2.keys()[0]

from epac import ds_split, ds_merge

leaf = wf.get_leftmost_leaf(); idx=0
leaf = wf.get_rightmost_leaf(); idx=-1

print leaf.get_key()
nodes = leaf.get_path_from_root().__iter__()
ds_kwargs = dict(X=X, y=y)

self = nodes.next()
print self.get_key(), "============================================"
if hasattr(self, "slices"):
    print self.slices
    print "Perm idx:", R2[key]['idx_perm'][idx]
    print "Train/text idx:", R2[key]['idx_train'][idx][idx], R2[key]['idx_test'][idx][idx]
if hasattr(self, "estimator"):
    ds_kwargs_train, ds_kwargs_test = ds_split(ds_kwargs)
    print "Test equality of input data"
    print np.all(ds_kwargs_train['X'] == R2[key]['X_train'][idx][idx])
    print np.all(ds_kwargs_test['X'] == R2[key]['X_test'][idx][idx])
    print "Stored results:", self.results
    print "Re-predict:", self.estimator.predict(ds_kwargs_test["X"])
    clf = LDA()
    clf.fit(ds_kwargs_train["X"], ds_kwargs_train["y"])
    print "New clf prediction", clf.predict(ds_kwargs_test["X"])
ds_kwargs = self.fit_predict(recursion=False, **ds_kwargs)


R2[key]['pred_y'][idx][idx]
R2[key]['X_train'][idx][idx]


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