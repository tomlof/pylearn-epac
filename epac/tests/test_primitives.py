# -*- coding: utf-8 -*-
"""
Created on Sun May 19 19:29:17 2013

@author: edouard.duchesnay@cea.fr

Test simple EPAC primitives.
"""

import unittest
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.lda import LDA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipe
from epac import CV, Methods, Permutations
from epac.sklearn_plugins import Permutation


class TestPipeline(unittest.TestCase):

    def test_pipeline(self):
        X, y = datasets.make_classification(n_samples=20, n_features=5,
                                            n_informative=2)
        # ===================
        # = With EPAC
        # ===================
        wf = Pipe(SelectKBest(k=2), SVC(kernel="linear"))
        wf.fit(X=X, y=y)
        r1 = wf.predict(X=X)
        # ===================
        # = Without EPAC
        # ===================
        pipe = Pipe([('anova', SelectKBest(k=2)),
                         ('svm', SVC(kernel="linear"))])
        pipe.fit(X, y)
        r2 = pipe.predict(X)
        self.assertTrue(np.all(r1 == r2), u'Diff Pipe')


class TestCV(unittest.TestCase):

    def test_cv(self):
        X, y = datasets.make_classification(n_samples=20, n_features=5,
                                            n_informative=2)
        n_folds = 2
        # ===================
        # = With EPAC
        # ===================
        wf = CV(SVC(kernel="linear"), n_folds=n_folds)
        wf.fit_predict(X=X, y=y)
        R1 = wf.reduce()
        # ===================
        # = Without EPAC
        # ===================
        from sklearn.cross_validation import StratifiedKFold
        clf = SVC(kernel="linear")
        R2 = list()
        for idx_train, idx_test in StratifiedKFold(y=y, n_folds=n_folds):
            #idx_train, idx_test  = cv.__iter__().next()
            X_train = X[idx_train, :]
            X_test = X[idx_test, :]
            y_train = y[idx_train, :]
            clf.fit(X_train, y_train)
            R2.append(clf.predict(X_test))
        comp = np.all(np.asarray(R1.values()[0]['pred_te']) == np.asarray(R2))
        self.assertTrue(comp, u'Diff CV')


class TestPermutations(unittest.TestCase):

    def test_perm(self):
        X, y = datasets.make_classification(n_samples=20, n_features=5,
                                            n_informative=2)
        n_perms = 2
        rnd = 0
        # ===================
        # = With EPAC
        # ===================
        wf = Permutations(SVC(kernel="linear"), n_perms=n_perms, permute="y", random_state=rnd)
        wf.fit_predict(X=X, y=y)
        R1 = wf.reduce()
        # ===================
        # = Without EPAC
        # ===================
        clf = SVC(kernel="linear")
        R2 = list()
        for perm in Permutation(n=y.shape[0], n_perms=n_perms,
                                               random_state=rnd):
            y_p = y[perm, :]
            clf.fit(X, y_p)
            R2.append(clf.predict(X))
        comp = np.all(np.asarray(R1.values()[0]['pred_te']) == np.asarray(R2))
        self.assertTrue(comp, u'Diff CV')

    def test_perm_cv(self):
            X, y = datasets.make_classification(n_samples=20, n_features=5,
                                                n_informative=2)
            n_perms = 2
            n_folds = 2
            rnd = 0
            # ===================
            # = With EPAC
            # ===================
            wf = Permutations(CV(SVC(kernel="linear"), n_folds=n_folds), n_perms=n_perms, permute="y", random_state=rnd)
            wf.fit_predict(X=X, y=y)
            R1 = wf.reduce()
            # ===================
            # = Without EPAC
            # ===================
            clf = SVC(kernel="linear")
            R2 = list()
            for perm in Permutation(n=y.shape[0], n_perms=n_perms,
                                                   random_state=rnd):
                y_p = y[perm, :]
                clf.fit(X, y_p)
                R2.append(clf.predict(X))
            comp = np.all(np.asarray(R1.values()[0]['pred_te']) == np.asarray(R2))
            self.assertTrue(comp, u'Diff CV')
class TestMethods(unittest.TestCase):
    
    def test_constructor_avoid_collision_level1(self):
        # Test that level 1 collisions are avoided
        pm = Methods(*[SVC(kernel="linear", C=C) for C in [1, 10]])
        leaves_key = [l.get_key() for l in pm.walk_leaves()]
        self.assertTrue(len(leaves_key) == len(set(leaves_key)),
                        u'Collision could not be avoided')

    def test_constructor_avoid_collision_level2(self):
        # Test that level 2 collisions are avoided
        pm = Methods(*[Pipe(SelectKBest(k=2), SVC(kernel="linear", C=C))\
                          for C in [1, 10]])
        leaves_key = [l.get_key() for l in pm.walk_leaves()]
        self.assertTrue(len(leaves_key) == len(set(leaves_key)),
                        u'Collision could not be avoided')

    def test_constructor_cannot_avoid_collision_level2(self):
        # This should raise an exception since collision cannot be avoided
        self.assertRaises(ValueError, Methods,
                         *[Pipe(SelectKBest(k=2), SVC(kernel="linear", C=C))\
                          for C in [1, 1]])

    def test_twomethods(self):
        X, y = datasets.make_classification(n_samples=20, n_features=5,
                                            n_informative=2)
        # ===================
        # = With EPAC
        # ===================
        wf = Methods(LDA(), SVC(kernel="linear"))
        wf.fit(X=X, y=y)
        r1 = wf.predict(X=X)
        #r1 ===================
        # = Without EPAC
        # ===================
        lda = LDA()
        svm = SVC(kernel="linear")
        lda.fit(X, y)
        svm.fit(X, y)
        r2 = [lda.predict(X), svm.predict(X)]
        comp = np.all(np.asarray(r1) == np.asarray(r2))
        self.assertTrue(comp, u'Diff Methods')

if __name__ == '__main__':
    unittest.main()