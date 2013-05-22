# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:37:54 2013

@author: edouard.duchesnay@cea.fr
"""


class Results(dict):
    TRAIN = "tr"
    TEST = "te"
    SCORE = "score"
    PRED = "pred"
    TRUE = "true"
    SEP = "_"

    def __init__(self, **kwargs):
        if kwargs:
            self.add(**kwargs)

    def _c(self, k1, k2):
        return k1 + self.SEP + k2

    def add(self, key2, suffix, score=None, pred=None, true=None):
        """
        Parameters
        ----------
        key2 str
            Secondary key
        
        suffix str
            Results.TRAIN or Results.TEST

        score Any type
            the score

        """
        if key2 in self:
            d = self[key2]
        else:
            d = dict()
            self[key2] = d
        if score is not None:
            d[self._c(self.SCORE, suffix)] = score
        if pred is not None:
            d[self._c(self.PRED, suffix)] = pred
        if true is not None:
            d[self._c(self.TRUE, suffix)] = true
        