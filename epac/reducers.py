# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 16:12:26 2013

Reducers for EPAC
@author: edouard.duchesnay@cea.fr
@author: benoit.da_mota@inria.fr
"""
import numpy as np
import re
from abc import abstractmethod
from epac import conf
from epac.results import Results
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
    def reduce(self, node, key2, result):
        """Reduce abstract method

        Parameters
        ----------
        node:
            the current workflow node that call the Reducer. Usefull only if
            one need to access workflow node, as in refit situations.

        key2:
            the intermediary key

        result: (dict)
            bag of results
        """


class SummaryStat(Reducer):
    """Reducer that select sub-result(s) according to select_regexp, and
    reduce the sub-result(s) using the statistics stat"""
    def __init__(self, do_stats_on_regexp=Results.SCORE, stat="mean",
                 filter_out_others=True):
        self.do_stats_on_regexp = do_stats_on_regexp
        self.stat = stat
        self.filter_out_others = filter_out_others

    def reduce(self, node, key2, result):
        if self.do_stats_on_regexp:
            select_keys = [key3 for key3 in result
                if re.search(self.do_stats_on_regexp, str(key3))]
        else:
            select_keys = result.keys()
        out = dict()
        for key3 in select_keys:
            if self.stat == "mean":
                out[self.stat + "_" + str(key3)] = \
                    np.mean(np.array(result[key3]), axis=0)
        if not self.filter_out_others:
            out.update(result)
        return out


class PvalPermutations(Reducer):
    """Reducer that select sub-result(s) according to select_regexp, and
    reduce the sub-result(s) using the statistics stat"""
    def __init__(self, select_regexp='mean.*' + Results.SCORE,
                 filter_out_others=True):
        self.select_regexp = select_regexp
        self.filter_out_others = filter_out_others

    def reduce(self,  node, key2, result):
        if self.select_regexp:
            select_keys = [key3 for key3 in result
                if re.search(self.select_regexp, str(key3))]
                #if re.search(self.select_regexp) != -1]
        else:
            select_keys = result.keys()
        out = dict()
        for key3 in select_keys:
            out[key3] = result[key3][0]
            count = np.sum(result[key3][1:] > result[key3][0])
            pval = count / (len(result[key3]) - 1)
            out["count_" + str(key3)] = count
            out["pval_" + str(key3)] = pval
        if not self.filter_out_others:
            out.update(result)
        return out