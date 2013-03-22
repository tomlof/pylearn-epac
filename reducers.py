# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 16:12:26 2013

Reducers for EPAC
@author: edouard.duchesnay@gmail.com
"""
import numpy as np 
import re
from abc import abstractmethod
from epac import conf

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


class SummaryStat(Reducer):
    
    """Reducer that select sub-result(s) according to select_regexp, and
    reduce the sub-result(s) using the statistics stat"""
    def __init__(self, do_stats_on_regexp=conf.PREFIX_SCORE, stat="mean",
                 filter_out_others=True):
        self.do_stats_on_regexp = do_stats_on_regexp
        self.stat = stat
        self.filter_out_others = filter_out_others
    def reduce(self, key2, result):
        out = dict()
        if self.do_stats_on_regexp:
            select_keys = [k for k in result
                if str(k).find(self.do_stats_on_regexp) != -1]
        else:
            select_keys = result.keys()
        for k in select_keys:
            if self.stat == "mean":
                out[self.stat + "_" + str(k)] = np.mean(result[k])
        if not self.filter_out_others:
            out.update(result)
        return out


class PvalPermutations(Reducer):
    """Reducer that select sub-result(s) according to select_regexp, and
    reduce the sub-result(s) using the statistics stat"""
    def __init__(self, select_regexp='mean.*'+conf.PREFIX_SCORE,
                 filter_out_others=True):
        self.select_regexp = select_regexp
        self.filter_out_others = filter_out_others

    def reduce(self, key2, result):
        out = dict()
        if self.select_regexp:
            select_keys = [k for k in result
                if re.search(self.select_regexp, str(k))]
                #if re.search(self.select_regexp) != -1]
        else:
            select_keys = result.keys()
        for k in select_keys:
            out[k] = result[k][0]
            count = np.sum(result[k][1:] > result[k][0])
            pval = count / (len(result[k]) - 1)
            out["count_" + str(k)] = count
            out["pval_" + str(k)] = pval
        if not self.filter_out_others:
            out.update(result)
        return out
