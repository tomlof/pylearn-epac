# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 16:12:26 2013

Reducers for EPAC
@author: edouard.duchesnay@gmail.com
"""
import numpy as np 
from abc import abstractmethod
from epac import Config

## ======================================================================== ##
## == Reducers                                                           == ##
## ======================================================================== ##


class Reducer(object):
    """ Reducer abstract class, called within the bottum_up method to process
    up-stream data flow of results.

    Inherited classes should implement reduce(key2, val). Where key2 is the
    intermediary key and val the corresponding results.
    This value is a dictionnary of results. The reduce should return a
    dictionnary."""
    @abstractmethod
    def reduce(self, key2, result):
        """Reduce abstract method

        Parameters
        ----------
        key2:
            the intermediary key

        result: (dict)
            bag of results
        """

class SelectAndDoStats(Reducer):
    """Reducer that select sub-result(s) according to select_regexp, and
    reduce the sub-result(s) using the statistics stat"""
    def __init__(self, select_regexp=Config.PREFIX_SCORE, stat="mean"):
        self.select_regexp = select_regexp
        self.stat = stat
    def reduce(self, key2, result):
        out = dict()
        if self.select_regexp:
            select_keys = [k for k in result
                if str(k).find(self.select_regexp) != -1]
        else:
            select_keys = result.keys()
        for k in select_keys:
            if self.stat == "mean":
                out[self.stat + "_" + str(k)] = np.mean(result[k])
        return out

class PvalPermutations(Reducer):
    """Reducer that select sub-result(s) according to select_regexp, and
    reduce the sub-result(s) using the statistics stat"""
    def __init__(self, select_regexp=Config.PREFIX_SCORE):
        self.select_regexp = select_regexp
    def reduce(self, key2, result):
        out = dict()
        if self.select_regexp:
            select_keys = [k for k in result
                if str(k).find(self.select_regexp) != -1]
        else:
            select_keys = result.keys()
        for k in select_keys:
            out[k] = result[k][0]
            count = np.sum(result[k][1:] > result[k][0])
            pval = count / (len(result[k]) - 1)
            out["count_" + str(k)] = count
            out["pval_" + str(k)] = pval
        return out
