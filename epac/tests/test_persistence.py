# -*- coding: utf-8 -*-
"""
Created on Sun May 19 20:32:20 2013

@author: edouard.duchesnay@cea.fr

Test persistence
"""

import unittest

import string
import os.path
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedKFold
from sklearn import grid_search

from epac import Pipe, Methods, CV, Perms
from epac import ClassificationReport, PvalPerms
from epac import StoreFs
from epac import CVBestSearchRefit
from epac.sklearn_plugins import Permutations
from epac.configuration import conf


class TestWorkFlow(unittest.TestCase):

    def test_peristence_load_and_fit_predict(self):
        X, y = datasets.make_classification(n_samples=20, n_features=10,
                                        n_informative=2)
        n_folds = 2
        n_folds_nested = 3
        k_values = [1, 2]
        C_values = [1, 2]
        pipelines = Methods(*[
                            Pipe(SelectKBest(k=k),
                            Methods(*[SVC(kernel="linear", C=C)
                            for C in C_values]))
                            for k in k_values])

        pipeline = CVBestSearchRefit(pipelines,
                                     n_folds=n_folds_nested)

        tree_mem = CV(pipeline, n_folds=n_folds,
                      reducer=ClassificationReport(keep=False))
        # Save Tree
        import tempfile
        store = StoreFs(dirpath=tempfile.mkdtemp(), clear=True)
        tree_mem.save_tree(store=store)
        tree_mem.run(X=X, y=y)
        res_mem = tree_mem.reduce().values()[0]
        # Reload Tree
        tree_fs_noresults = store.load()
        tree_fs_noresults.run(X=X, y=y)
        res_fs_noresults = tree_fs_noresults.reduce().values()[0]
        # Save with results
        tree_fs_noresults.save_tree(store=store)
        tree_fs_withresults = store.load()
        res_fs_withresults = tree_fs_withresults.reduce().values()[0]
        #
        # Compare
        comp = np.all([
            np.all(
            np.asarray(res_mem[k]) == np.asarray(res_fs_noresults[k]))
            and
            np.all(np.asarray(res_fs_noresults[k]) ==
            np.asarray(res_fs_withresults[k]))
            for k in res_mem])
        self.assertTrue(comp)

    def test_peristence_perm_cv_parmethods_pipe_vs_sklearn(self):
        key_y_pred = 'y' + conf.SEP + conf.PREDICTION
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
        # Save workflow
        # -------------
        import tempfile
        #store = StoreFs("/tmp/toto", clear=True)
        store = StoreFs(tempfile.mktemp())
        wf.save_tree(store=store)
        wf = store.load()
        wf.run(X=X, y=y)
        ## Save results
        wf.save_tree(store=store)
        wf = store.load()
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
        r_sklearn[conf.BEST_PARAMS]['name'] = 'SVC'

        # - Comparisons
        comp = np.all(r_epac[key_y_pred] == r_sklearn[key_y_pred])
        self.assertTrue(comp, u'Diff CVBestSearchRefit: prediction')
        comp = np.all([r_epac[conf.BEST_PARAMS][0][p] == r_sklearn[conf.BEST_PARAMS][p]
        for p in  r_sklearn[conf.BEST_PARAMS]])
        self.assertTrue(comp, u'Diff CVBestSearchRefit: best parameters')


if __name__ == '__main__':
    unittest.main()