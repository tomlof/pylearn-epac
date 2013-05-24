# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:37:54 2013

@author: edouard.duchesnay@cea.fr
"""


class Results(dict):
    """Results is a dictionnary indexed by intermediry keys (key2). It contains
    Result.

    See also
    --------
    Result
    """

    def __init__(self, **kwargs):
        if kwargs:
            self.add(**kwargs)

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
            res = self[key2]
        else:
            res = Result()
            self[key2] = res
        if score is not None:
            res[Result.concat_key3(Result.SCORE, suffix)] = score
        if pred is not None:
            res[Result.concat_key3(Result.PRED, suffix)] = pred
        if true is not None:
            res[Result.concat_key3(Result.TRUE, suffix)] = true


class Result(dict):
    """Result is a dictionnary indexed by tertiary keys (key3).

    See also
    --------
    Result
    """

    TRAIN = "tr"
    TEST = "te"
    SCORE = "score"
    PRED = "pred"
    TRUE = "true"
    SEP = "_"

    @classmethod
    def concat_key3(self, k1, k2):
        """Concatenate keys 3"""
        return k1 + self.SEP + k2
