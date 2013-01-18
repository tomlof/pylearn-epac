# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 16:20:03 2012

@author: edouard
"""

## Class Permutation to be added to sklearn
import numpy as np
from sklearn.utils import check_random_state


class Permutation(object):

    def __init__(self, n, n_perms, first_perm_is_id=True, random_state=None):
        self.random_state = random_state
        self.first_perm_is_id = first_perm_is_id
        if abs(n - int(n)) >= np.finfo('f').eps:
            raise ValueError("n must be an integer")
        self.n = int(n)
        if abs(n_perms - int(n_perms)) >= np.finfo('f').eps:
            raise ValueError("n_perms must be an integer")
        self.n_perms = int(n_perms)

    def __iter__(self):
        rng = check_random_state(self.random_state)
        if self.first_perm_is_id:
            yield np.arange(self.n)  # id permutation
            for i in xrange(self.n_perms - 1): # n_perms-1 random permutations
                yield rng.permutation(self.n)
        else:
            for i in xrange(self.n_perms):  # n_perms random permutations
                yield rng.permutation(self.n)

    def __repr__(self):
        return '%s.%s(n=%i)' % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.n,
        )

    def __len__(self):
        return self.n_perms

from sklearn.feature_selection import SelectKBest

class FeatureRanking():

    def __init__(self, score_func=f_classif):
        """ Initialize the univariate feature ranking.

        Parameters
        ----------
        score_func : callable
            Function taking two arrays X and y, and returning a pair of arrays
            (scores, pvalues).
        """
        if not callable(score_func):
            raise TypeError(
                "The score function should be a callable, %s (%s) "
                "was passed." % (score_func, type(score_func)))
        self.score_func = score_func

    def fit(self, X, y):
        """
        Evaluate the function
        """
        self.scores_, self.pvalues_ = self.score_func(X, y)
        if len(np.unique(self.pvalues_)) < len(self.pvalues_):
            warn("Duplicate p-values. Result may depend on feature ordering."
                 "There are probably duplicate features, or you used a "
                 "classification score for a regression task.")
        return self
