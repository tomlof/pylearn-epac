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
import sklearn.pipeline
from sklearn.cross_validation import StratifiedKFold
from sklearn import grid_search
from epac import Pipe, Methods, CV, Perms, CVBestSearchRefit
from epac import ClassificationReport
from epac.sklearn_plugins import Permutations
from epac.configuration import conf


class TestPipeline(unittest.TestCase):

    def test_pipeline(self):
        X, y = datasets.make_classification(n_samples=20,
                                            n_features=5,
                                            n_informative=2)
        # = With EPAC
        wf = Pipe(SelectKBest(k=2), SVC(kernel="linear"))
        r_epac = wf.top_down(X=X, y=y)

        # = With SKLEARN
        pipe = sklearn.pipeline.Pipeline([('anova', SelectKBest(k=2)),
                         ('svm', SVC(kernel="linear"))])
        r_sklearn = pipe.fit(X, y).predict(X)

        key2cmp = 'y' + conf.SEP + conf.PREDICTION
        # = Comparison
        self.assertTrue(np.all(r_epac[key2cmp] == r_sklearn),
                        u'Diff in Pipe: EPAC vs sklearn')
        # test reduce
        r_epac_reduce = wf.reduce().values()[0][key2cmp]
        self.assertTrue(np.all(r_epac_reduce == r_sklearn),
                        u'Diff in Pipe: EPAC reduce')


class TestCV(unittest.TestCase):

    def test_cv(self):
        X, y = datasets.make_classification(n_samples=20, n_features=5,
                                            n_informative=2)
        n_folds = 2

        # = With EPAC
        wf = CV(SVC(kernel="linear"), n_folds=n_folds,
                reducer=ClassificationReport(keep=True))
        r_epac = wf.top_down(X=X, y=y)

        # = With SKLEARN
        clf = SVC(kernel="linear")
        r_sklearn = list()
        for idx_train, idx_test in StratifiedKFold(y=y, n_folds=n_folds):
            #idx_train, idx_test  = cv.__iter__().next()
            X_train = X[idx_train, :]
            X_test = X[idx_test, :]
            y_train = y[idx_train, :]
            clf.fit(X_train, y_train)
            r_sklearn.append(clf.predict(X_test))

        # = Comparison
        key2cmp = 'y' + conf.SEP + conf.TEST + conf.SEP + conf.PREDICTION
        for icv in range(n_folds):
            comp = np.all(np.asarray(r_epac[0][key2cmp]) \
                                    == np.asarray(r_sklearn[0]))
            self.assertTrue(comp, u'Diff CV: EPAC vs sklearn')

        # test reduce
        r_epac_reduce = wf.reduce().values()[0][key2cmp]
        comp = np.all(np.asarray(r_epac_reduce) == np.asarray(r_sklearn))
        self.assertTrue(comp, u'Diff CV: EPAC reduce')


class TestPerms(unittest.TestCase):

    def test_perm(self):
        X, y = datasets.make_classification(n_samples=20,
                                            n_features=5,
                                            n_informative=2)
        n_perms = 2
        rnd = 0
        # = With EPAC
        wf = Perms(SVC(kernel="linear"), n_perms=n_perms, permute="y",
                          random_state=rnd, reducer=None)
        r_epac = wf.top_down(X=X, y=y)
        # = With SKLEARN
        clf = SVC(kernel="linear")
        r_sklearn = list()
        for perm in Permutations(n=y.shape[0], n_perms=n_perms,
                                 random_state=rnd):
            y_p = y[perm, :]
            clf.fit(X, y_p)
            r_sklearn.append(clf.predict(X))
        key2cmp = 'y' + conf.SEP + conf.PREDICTION

        # = Comparison
        for iperm in range(n_perms):
            comp = np.all(
                    np.asarray(r_epac[iperm][key2cmp])
                    ==
                    np.asarray(r_sklearn[iperm]))
            self.assertTrue(comp, u'Diff Perm: EPAC vs sklearn')
        # test reduce
        for iperm in range(n_perms):
            r_epac_reduce = wf.reduce().values()[iperm][key2cmp]
            comp = np.all(np.asarray(r_epac_reduce) 
                          == np.asarray(r_sklearn[iperm]))
            self.assertTrue(comp, u'Diff Perm: EPAC reduce')
    
    def test_perm2(self):
        from epac.tests.wfexamples2test import WFExample2
        X, y = datasets.make_classification(n_samples=20, n_features=5,
                                            n_informative=2)
        wf = WFExample2().get_workflow()
        wf.run(X=X, y=y)
        wf.reduce()


class TestCVBestSearchRefit(unittest.TestCase):

    def test_cvbestsearchrefit(self):
        X, y = datasets.make_classification(n_samples=12,
                                            n_features=10,
                                            n_informative=2)
        n_folds_nested = 2
        #random_state = 0
        C_values = [.1, 0.5, 1, 2, 5]
        kernels = ["linear", "rbf"]
        key_y_pred = 'y' + conf.SEP + conf.PREDICTION
        # With EPAC
        methods = Methods(*[SVC(C=C, kernel=kernel)
            for C in C_values for kernel in kernels])
        wf = CVBestSearchRefit(methods, n_folds=n_folds_nested)
        wf.run(X=X, y=y)
        r_epac = wf.reduce().values()[0]
        # - Without EPAC
        r_sklearn = dict()
        clf = SVC(kernel="linear")
        parameters = {'C': C_values, 'kernel': kernels}
        cv_nested = StratifiedKFold(y=y, n_folds=n_folds_nested)
        gscv = grid_search.GridSearchCV(clf, parameters, cv=cv_nested)
        gscv.fit(X, y)
        r_sklearn[key_y_pred] = gscv.predict(X)
        r_sklearn[conf.BEST_PARAMS] = gscv.best_params_
        # - Comparisons
        comp = np.all(r_epac[key_y_pred] == r_sklearn[key_y_pred])
        self.assertTrue(comp, u'Diff CVBestSearchRefit: prediction')
        for key_param in r_epac[conf.BEST_PARAMS][0]:
            if key_param in r_sklearn[conf.BEST_PARAMS]:
                comp = r_sklearn[conf.BEST_PARAMS][key_param] == \
                        r_epac[conf.BEST_PARAMS][0][key_param]
                self.assertTrue(comp, \
                    u'Diff CVBestSearchRefit: best parameters')

    def test_cvbestsearchrefit_select_k_best(self):
        list_C_value = range(2, 10, 1)
#        print repr(list_C_value)
        for C_value in list_C_value:
#            C_value = 2
#            print C_value
            X, y = datasets.make_classification(n_samples=100,
                                                n_features=500,
                                                n_informative=5)
            n_folds_nested = 2
            #random_state = 0
            k_values = [2, 3, 4, 5, 6]
            key_y_pred = 'y' + conf.SEP + conf.PREDICTION
            # With EPAC
            methods = Methods(*[Pipe(SelectKBest(k=k),
                                     SVC(C=C_value, kernel="linear"))
                                     for k in k_values])
            wf = CVBestSearchRefit(methods, n_folds=n_folds_nested)
            wf.run(X=X, y=y)
            r_epac = wf.reduce().values()[0]
            # - Without EPAC
            from sklearn.pipeline import Pipeline
            r_sklearn = dict()
            clf = Pipeline([('anova', SelectKBest(k=3)),
                            ('svm', SVC(C=C_value, kernel="linear"))])
            parameters = {'anova__k': k_values}
            cv_nested = StratifiedKFold(y=y, n_folds=n_folds_nested)
            gscv = grid_search.GridSearchCV(clf, parameters, cv=cv_nested)
            gscv.fit(X, y)
            r_sklearn[key_y_pred] = gscv.predict(X)
            r_sklearn[conf.BEST_PARAMS] = gscv.best_params_
            r_sklearn[conf.BEST_PARAMS]['k'] = \
                r_sklearn[conf.BEST_PARAMS]['anova__k']
            # - Comparisons
            comp = np.all(r_epac[key_y_pred] == r_sklearn[key_y_pred])
            self.assertTrue(comp, u'Diff CVBestSearchRefit: prediction')
            for key_param in r_epac[conf.BEST_PARAMS][0]:
                if key_param in r_sklearn[conf.BEST_PARAMS]:
                    comp = r_sklearn[conf.BEST_PARAMS][key_param] == \
                            r_epac[conf.BEST_PARAMS][0][key_param]
                    self.assertTrue(comp, \
                        u'Diff CVBestSearchRefit: best parameters')


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
        key_y_pred = 'y' + conf.SEP + conf.PREDICTION
        X, y = datasets.make_classification(n_samples=20, n_features=5,
                                            n_informative=2)
        # = With EPAC
        wf = Methods(LDA(), SVC(kernel="linear"))
        r_epac = wf.run(X=X, y=y)

        # = With SKLEARN
        lda = LDA()
        svm = SVC(kernel="linear")
        lda.fit(X, y)
        svm.fit(X, y)
        r_sklearn = [lda.predict(X), svm.predict(X)]

        # Comparison
        for i_cls in range(2):
            comp = np.all(np.asarray(r_epac[i_cls][key_y_pred]) ==
                                    np.asarray(r_sklearn[i_cls]))
            self.assertTrue(comp, u'Diff Methods')

        # test reduce
        r_epac_reduce = [wf.reduce().values()[0][key_y_pred],
            wf.reduce().values()[1][key_y_pred]]
        comp = np.all(np.asarray(r_epac_reduce) == np.asarray(r_sklearn))
        self.assertTrue(comp, u'Diff Perm / CV: EPAC reduce')

if __name__ == '__main__':
    unittest.main()