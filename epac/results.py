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

    def add(self, key2, suffix, score=None, pred=[], true=[]):
        tr = dict()
        if score:
            tr[self._c(self.SCORE, suffix)] = score
        if len(pred):
            tr[self._c(self.PRED, suffix)] = pred
        if len(true):
            tr[self._c(self.TRUE, suffix)] = true
        if key2 in self:
            self[key2].update(tr)
        else:
            self[key2] = tr