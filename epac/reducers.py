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
from epac.configuration import conf
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
    def reduce(self, result):
        """Reduce abstract method

        Parameters
        ----------
        key2:
            the intermediary key

        result: (dict)
            bag of results
        """


class SummaryStat(Reducer):
    """Reducer that select sub-result(s) items according to select_regexp, and
    reduce the sub-result(s) using the statistics stat.
    
    select_regexp: srt
      A string to select items (defaults %s)

    keep: boolean
      Should other items be kept (False) into summarized results.
      (default False)

    Example:
    >>> from epac import SummaryStat
    >>> result = {'score_tr': [1, .8], 'score_te': [.9, .7]}
    >>> print SummaryStat(keep=True).reduce(result)
    {'score_te': [0.9, 0.7], 'mean_score_te': 0.80000000000000004, 'mean_score_tr': 0.90000000000000002, 'score_tr': [1, 0.8]}
    >>> print SummaryStat(keep=False).reduce(result)
    {'mean_score_te': 0.80000000000000004, 'mean_score_tr': 0.90000000000000002}
    """ % Results.SCORE
    def __init__(self, select_regexp=Results.SCORE, stat="mean",
                 keep=False):
        self.select_regexp = select_regexp
        self.stat = stat
        self.keep = keep

    def reduce(self, result):
        if self.select_regexp:
            select_key3s = [key3 for key3 in result
                if re.search(self.select_regexp, str(key3))]
        else:
            select_key3s = result.keys()
        out = Results()
        for key3 in select_key3s:
            if self.stat == "mean":
                out[Results.concat_key3(self.stat, str(key3))] = \
                    np.mean(np.array(result[key3]), axis=0)
        if self.keep:
            out.update(result)
        return out


class PvalPermutations(Reducer):
    """Reducer that select sub-result(s) according to select_regexp, and
    reduce the sub-result(s) using the statistics stat"""
    def __init__(self, select_regexp='mean.*' + Results.SCORE,
                 keep=False):
        self.select_regexp = select_regexp
        self.keep = keep

    def reduce(self, result):
        if self.select_regexp:
            select_keys = [key3 for key3 in result
                if re.search(self.select_regexp, str(key3))]
                #if re.search(self.select_regexp) != -1]
        else:
            select_keys = result.keys()
        out = Results()
        for key3 in select_keys:
            out[key3] = result[key3][0]
            count = np.sum(result[key3][1:] > result[key3][0])
            pval = count / (len(result[key3]) - 1)
            out["count_" + str(key3)] = count
            out["pval_" + str(key3)] = pval
        if self.keep:
            out.update(result)
        return out