# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 16:20:03 2012

@author: edouard.duchesnay@cea.fr
"""

## Class Permutations to be added to sklearn
import numpy as np
from sklearn.utils import check_random_state


class Permutations(object):

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
            for i in xrange(self.n_perms - 1): # n_perms-1 random Permutationss
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

def _clean_nans(scores):
    """
    NaNs can't be properly compared, so change them to the
    smallest value of scores's dtype. -inf seems to be unreliable.
    """
    # XXX where should this function be called? fit? scoring functions
    # themselves?
    scores[np.isnan(scores)] = np.finfo(scores.dtype).min
    return scores

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

class FeatureRanking():

    def __init__(self, score_func=f_classif):
        """ Initialize the univariate feature ranking.

        Parameters
        ----------
        score_func : callable
            Function taking two arrays X and y, and returning a pair of arrays
            (scores, pvalues).
            
        Example
        -------
        from sklearn import datasets
        import numpy as np
        # import some data to play with
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        Xn = np.random.normal(size=X.shape[0]*100).reshape((X.shape[0], 100))
        X = np.hstack((X, Xn))
        filter = FeatureRanking()
        filter.fit(X, y)
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
        self.scores, self.pvalues = self.score_func(X, y)
        self.ranks = np.argsort(self.scores)[::-1]
        if len(np.unique(self.pvalues)) < len(self.pvalues):
            warn("Duplicate p-values. Result may depend on feature ordering."
                 "There are probably duplicate features, or you used a "
                 "classification score for a regression task.")
        return self

    def transform(self, X):
        return X

    def toto(self):
        return dict(fscores=self.scores, pvalues=self.pvalues, ranks=self.ranks)