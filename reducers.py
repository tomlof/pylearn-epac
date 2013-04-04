# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 16:12:26 2013

Reducers for EPAC
@author: edouard.duchesnay@gmail.com
"""
import copy
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
    def __init__(self, do_stats_on_regexp=conf.PREFIX_SCORE, stat="mean",
                 filter_out_others=True):
        self.do_stats_on_regexp = do_stats_on_regexp
        self.stat = stat
        self.filter_out_others = filter_out_others

    def reduce(self,  node, key2, result):
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


class CVGridSearchRefit(Reducer):
    """Reducer that select sub-result(s) according to select_regexp, and
    reduce the sub-result(s) using the statistics stat"""

    def __init__(self, key3="test.+" + conf.PREFIX_SCORE,
                 arg_max=True):
        self.key3 = key3
        self.arg_max = arg_max

    def reduce(self, node, key2, result):
        #print node, key2, result
        match_key3 = [key3 for key3 in result
            if re.search(self.key3, str(key3))]
        if len(match_key3) != 1:
            raise ValueError("None or more than one tertiary key found")
        # 1) Retrieve pairs of optimal (argument-name, value)
        key3 = match_key3[0]
        grid_cv = result[key3]
        mean_cv = np.mean(np.array(grid_cv), axis=0)
        mean_cv_opt = np.max(mean_cv) if self.arg_max else  np.min(mean_cv)
        idx_opt = np.where(mean_cv == mean_cv_opt)
        idx_opt = [idx[0] for idx in idx_opt]
        grid = grid_cv[0]
        args_opt = list()
        while len(idx_opt):
            idx = idx_opt[0]
            args_opt.append((grid.axis_name, grid.axis_values[idx]))
            idx_opt = idx_opt[1:]
            grid = grid[0]
        #args_opt
        # Retrieve one node that match intermediary key
        leaf = node.get_node(regexp=key2, stop_first_match=True)
        # get path from current node
        path = leaf.get_path_from_node(node)
        # Strip of non estimator nodes
        path = [copy.deepcopy(n) for n in path if hasattr(n, "estimator")]
        re_argnames = re.compile(u'([\w]+)=[\w]')
        for n in path:
            n.signature_args
            args_opt
        path = Seq(*path)
        
        re_argnames.findall(n.get_signature())
        return out

# self=node


class PvalPermutations(Reducer):
    """Reducer that select sub-result(s) according to select_regexp, and
    reduce the sub-result(s) using the statistics stat"""
    def __init__(self, select_regexp='mean.*' + conf.PREFIX_SCORE,
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