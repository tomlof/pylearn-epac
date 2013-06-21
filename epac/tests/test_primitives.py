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
from epac import SummaryStat
from epac.sklearn_plugins import Permutations

class TestPipeline(unittest.TestCase):

    def test_pipeline(self):
        X, y = datasets.make_classification(n_samples=20, n_features=5,
                                            n_informative=2)

        # = With EPAC
        wf = Pipe(SelectKBest(k=2), SVC(kernel="linear"))
        r_epac = wf.fit_predict(X=X, y=y)

        # = With SKLEARN
        pipe = sklearn.pipeline.Pipeline([('anova', SelectKBest(k=2)),
                         ('svm', SVC(kernel="linear"))])
        r_sklearn = pipe.fit(X, y).predict(X)

        # = Comparison
        self.assertTrue(np.all(r_epac == r_sklearn),
                        u'Diff in Pipe: EPAC vs sklearn')
        # test reduce
        r_epac_reduce = wf.reduce().values()[0]['pred_te']
        self.assertTrue(np.all(r_epac_reduce == r_sklearn),
                        u'Diff in Pipe: EPAC reduce')


class TestCV(unittest.TestCase):

    def test_cv(self):
        X, y = datasets.make_classification(n_samples=20, n_features=5,
                                            n_informative=2)
        n_folds = 2

        # = With EPAC
        wf = CV(SVC(kernel="linear"), n_folds=n_folds,
                reducer=SummaryStat(keep=True))
        r_epac = wf.fit_predict(X=X, y=y)

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
        comp = np.all(np.asarray(r_epac) == np.asarray(r_sklearn))
        self.assertTrue(comp, u'Diff CV: EPAC vs sklearn')

        # test reduce
        r_epac_reduce = wf.reduce().values()[0]['pred_te']
        comp = np.all(np.asarray(r_epac_reduce) == np.asarray(r_sklearn))
        self.assertTrue(comp, u'Diff CV: EPAC reduce')


class TestPerms(unittest.TestCase):

    def test_perm(self):
        X, y = datasets.make_classification(n_samples=20, n_features=5,
                                            n_informative=2)
        n_perms = 2
        rnd = 0

        # = With EPAC
        wf = Perms(SVC(kernel="linear"), n_perms=n_perms, permute="y",
                          random_state=rnd, reducer=None)
        r_epac = wf.fit_predict(X=X, y=y)

        # = With SKLEARN
        clf = SVC(kernel="linear")
        r_sklearn = list()
        for perm in Permutations(n=y.shape[0], n_perms=n_perms,
                                 random_state=rnd):
            y_p = y[perm, :]
            clf.fit(X, y_p)
            r_sklearn.append(clf.predict(X))

        # = Comparison
        comp = np.all(np.asarray(r_epac) == np.asarray(r_sklearn))
        self.assertTrue(comp, u'Diff Perm: EPAC vs sklearn')

        # test reduce
        r_epac_reduce = [v['pred_te'] for v in wf.reduce().values()]
        comp = np.all(np.asarray(r_epac_reduce) == np.asarray(r_sklearn))
        self.assertTrue(comp, u'Diff Perm: EPAC reduce')


class TestCVBestSearchRefit(unittest.TestCase):

    def test_cvbestsearchrefit(self):
        X, y = datasets.make_classification(n_samples=12, n_features=10,
                                            n_informative=2)
        n_folds_nested = 2
        #random_state = 0
        C_values = [.1, 0.5, 1, 2, 5]
        kernels = ["linear", "rbf"]
        # With EPAC
        methods = Methods(*[SVC(C=C, kernel=kernel)
            for C in C_values for kernel in kernels])
        wf = CVBestSearchRefit(methods, n_folds=n_folds_nested)
        wf.fit_predict(X=X, y=y)
        r_epac = wf.reduce().values()[0]

        # - Without EPAC
        r_sklearn = dict()
        clf = SVC(kernel="linear")
        parameters = {'C': C_values, 'kernel': kernels}
        cv_nested = StratifiedKFold(y=y, n_folds=n_folds_nested)
        gscv = grid_search.GridSearchCV(clf, parameters, cv=cv_nested)
        gscv.fit(X, y)
        r_sklearn['pred_te'] = gscv.predict(X)
        r_sklearn['best_params'] = gscv.best_params_

        # - Comparisons
        comp = np.all(r_epac['pred_te'] == r_sklearn['pred_te'])
        self.assertTrue(comp, u'Diff CVBestSearchRefit: prediction')
        comp = np.all([r_epac['best_params'][0][p] == \
                       r_sklearn['best_params'][p]
        for p in  r_sklearn['best_params']])
        self.assertTrue(comp, u'Diff CVBestSearchRefit: best parameters')

    def test_cvbestsearchrefit_select_k_best(self):
        list_C_value = range(2, 10, 1)
#        print repr(list_C_value)
        for C_value in list_C_value:
#            print C_value
            X, y = datasets.make_classification(n_samples=100,
                                                n_features=500,
                                                n_informative=5)
            n_folds_nested = 2
            #random_state = 0
            k_values = [2, 3, 4, 5, 6]
            # With EPAC
            methods = Methods(*[Pipe(SelectKBest(k=k),
                                     SVC(C=C_value, kernel="linear"))
                                     for k in k_values])
            wf = CVBestSearchRefit(methods, n_folds=n_folds_nested)
            wf.fit_predict(X=X, y=y)
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
            r_sklearn['pred_te'] = gscv.predict(X)
            r_sklearn['best_params'] = gscv.best_params_
            # - Comparisons
            comp = np.all(r_epac['pred_te'] == r_sklearn['pred_te'])
            self.assertTrue(comp, u'Diff CVBestSearchRefit: prediction')
            for p in r_sklearn['best_params']:
                for p2 in r_epac['best_params'][0]:
                    if p2 in p:
                        r_epac['best_params'][0][p] = \
                            r_epac['best_params'][0][p2]
                        del r_epac['best_params'][0][p2]
                        break
            comp = np.all([r_epac['best_params'][0][p] == \
                r_sklearn['best_params'][p]
              for p in  r_sklearn['best_params']])
            self.assertTrue(comp, u'Diff CVBestSearchRefit: best parameters')

    def test_cvbestsearchrefit_select_k_best_with_C(self):
        X, y = datasets.make_classification(n_samples=100, n_features=500,
                                            n_informative=5)
        n_folds_nested = 2
        #random_state = 0
        k_values = [2, 3, 4, 5, 6]
        C_values = range(1, 10, 1)
        # With EPAC
        methods = Methods(*[Pipe(SelectKBest(k=k),
                                 SVC(C=C, kernel="linear"))
                                 for k in k_values
                                 for C in C_values])
        wf = CVBestSearchRefit(methods, n_folds=n_folds_nested)
        wf.fit_predict(X=X, y=y)
        r_epac = wf.reduce().values()[0]

        # - Without EPAC
        from sklearn.pipeline import Pipeline
        r_sklearn = dict()
        clf = Pipeline([('anova', SelectKBest(k=3)),
                        ('svm', SVC(C=1, kernel="linear"))])
        parameters = {'anova__k': k_values, 'svm__C': C_values}
        cv_nested = StratifiedKFold(y=y, n_folds=n_folds_nested)
        gscv = grid_search.GridSearchCV(clf, parameters, cv=cv_nested)
        gscv.fit(X, y)
        r_sklearn['pred_te'] = gscv.predict(X)
        r_sklearn['best_params'] = gscv.best_params_

        # - Comparisons
        comp = np.all(r_epac['pred_te'] == r_sklearn['pred_te'])
        self.assertTrue(comp, u'Diff CVBestSearchRefit: prediction')
        best_params_epac = {}
        for p in r_sklearn['best_params']:
            for res_epac in r_epac['best_params']:
                for p2 in res_epac:
                    if p2 in p:
                        best_params_epac[p] = res_epac[p2]
                        break
                if p in best_params_epac:
                    break
        comp = np.all([best_params_epac[p] == r_sklearn['best_params'][p]
          for p in  r_sklearn['best_params']])
        self.assertTrue(comp, u'Diff CVBestSearchRefit: best parameters')

    def test_cvbestsearchrefit_select_k_best_with_C_perm(self):
        nb = range(1, 2)
        for n in nb:
            X, y = datasets.make_classification(n_samples=100, n_features=500,
                                                n_informative=5)
            n_folds_nested = 2
            #random_state = 0
            k_values = [2, 3, 4, 5, 6]
            C_values = range(1, 10, 1)
            # With EPAC
            methods = Methods(*[Pipe(SelectKBest(k=k),
                                     SVC(C=C, kernel="linear"))
                                     for C in C_values
                                     for k in k_values])
            wf = CVBestSearchRefit(methods, n_folds=n_folds_nested)
            wf.fit_predict(X=X, y=y)
            r_epac = wf.reduce().values()[0]
            # - Without EPAC
            from sklearn.pipeline import Pipeline
            r_sklearn = dict()
            clf = Pipeline([('anova', SelectKBest(k=3)),
                            ('svm', SVC(C=1, kernel="linear"))])
            parameters = {'anova__k': k_values, 'svm__C': C_values}
            cv_nested = StratifiedKFold(y=y, n_folds=n_folds_nested)
            gscv = grid_search.GridSearchCV(clf, parameters, cv=cv_nested)
            gscv.fit(X, y)
            r_sklearn['pred_te'] = gscv.predict(X)
            r_sklearn['best_params'] = gscv.best_params_
            # - Comparisons
            best_params_epac = {}
            for p in r_sklearn['best_params']:
                for res_epac in r_epac['best_params']:
                    for p2 in res_epac:
                        if p2 in p:
                            best_params_epac[p] = res_epac[p2]
                            break
                    if p in best_params_epac:
                        break
            comp = np.all(r_epac['pred_te'] == r_sklearn['pred_te'])
            if not comp:
                print "r_epac (pred_te) =" + repr(r_epac['pred_te'])
                print "r_sklearn (pred_te) =" + repr(r_sklearn['pred_te'])
                print "r_epac (best_params) =" + repr(best_params_epac)
                print "r_sklearn (best_params) =" + \
                    repr(r_sklearn['best_params'])
            self.assertTrue(comp, u'Diff CVBestSearchRefit: prediction')
            comp = np.all([best_params_epac[p] == r_sklearn['best_params'][p]
              for p in  r_sklearn['best_params']])
            self.assertTrue(comp, u'Diff CVBestSearchRefit: best parameters')


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
        # = With EPAC
        wf = Methods(LDA(), SVC(kernel="linear"))
        r_epac = wf.fit_predict(X=X, y=y)

        # = With SKLEARN
        lda = LDA()
        svm = SVC(kernel="linear")
        lda.fit(X, y)
        svm.fit(X, y)
        r_sklearn = [lda.predict(X), svm.predict(X)]

        # Comparison
        comp = np.all(np.asarray(r_epac) == np.asarray(r_sklearn))
        self.assertTrue(comp, u'Diff Methods')

        # test reduce
        r_epac_reduce = [wf.reduce().values()[0]['pred_te'],
            wf.reduce().values()[1]['pred_te']]
        comp = np.all(np.asarray(r_epac_reduce) == np.asarray(r_sklearn))
        self.assertTrue(comp, u'Diff Perm / CV: EPAC reduce')

if __name__ == '__main__':
    unittest.main()