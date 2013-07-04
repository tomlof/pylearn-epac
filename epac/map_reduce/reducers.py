# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 16:12:26 2013

Reducers for EPAC
@author: edouard.duchesnay@cea.fr
"""
import numpy as np
import re
from abc import abstractmethod
from epac.map_reduce.results import Result
from epac.configuration import conf
from epac.workflow.base import key_push, key_pop
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

## ======================================================================== ##
## == Reducers                                                           == ##
## ======================================================================== ##


class Reducer(object):
    """ Reducer abstract class, called within the reduce method to process
    up-stream data flow of Result.

    Inherited classes should implement reduce(result)."""
    @abstractmethod
    def reduce(self, result):
        """Reduce abstract method

        Parameters
        ----------

        result: (dict)
            A result
        """


class ClassificationReport(Reducer):
    """Reducer compute classification statistics.

    select_regexp: srt
      A string to select items (defaults "test"). It must match two items:
      "true/test" and "pred/test".

    keep: boolean
      Should other items be kept (False) into summarized results.
      (default False)

    Example:
    >>> from epac import ClassificationReport
    >>> reducer = ClassificationReport()
    >>> reducer.reduce({'key': "SVC", 'y/test/pred': [0, 1, 1, 1], 'y/test/true': [0, 0, 1, 1]})
    {'key': SVC, 'y/test/score_accuray': 0.75, 'y/test/score_precision': [ 1.          0.66666667], 'y/test/score_recall': [ 0.5  1. ], 'y/test/score_f1': [ 0.66666667  0.8       ], 'y/test/score_recall_mean': 0.75}

    """
    def __init__(self, select_regexp=conf.TEST,
                 keep=False):
        self.select_regexp = select_regexp
        self.keep = keep

    def reduce(self, result):
        if self.select_regexp:
            inputs = [key3 for key3 in result
                if re.search(self.select_regexp, str(key3))]
        else:
            inputs = result.keys()
        if len(inputs) != 2:
            raise KeyError("Need to find exactly two result to compute a score."
            "Found %i: %s" % (len(inputs), inputs))
        key_true = [k for k in inputs if k.find(conf.TRUE) != -1][0]
        key_pred = [k for k in inputs if k.find(conf.PREDICTION) != -1][0]
        y_true = result[key_true]
        y_pred = result[key_pred]
        try:  # If list of arrays (CV, LOO, etc.) concatenate them
            y_true = np.concatenate(y_true)
            y_pred = np.concatenate(y_pred)
        except ValueError:
            pass
        out = Result(key=result["key"])
        p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, average=None)
        key, _ = key_pop(key_pred, -1)
        out[key_push(key, conf.SCORE_PRECISION)] = p
        out[key_push(key, conf.SCORE_RECALL)] = r
        out[key_push(key, conf.SCORE_RECALL_MEAN)] = r.mean()
        out[key_push(key, conf.SCORE_F1)] = f1
        out[key_push(key, conf.SCORE_ACCURACY)] = accuracy_score(y_true, y_pred)
        if self.keep:
            out.update(result)
        return out


class PvalPerms(Reducer):
    """Reducer that select sub-result(s) according to select_regexp, and
    reduce the sub-result(s) using the statistics stat"""
    def __init__(self, select_regexp='mean.*' + Result.SCORE,
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
        out = Result(key=result.key())
        for key3 in select_keys:
            out[key3] = result[key3][0]
            count = np.sum(result[key3][1:] > result[key3][0])
            pval = float(count) / (len(result[key3]) - 1)
            #out[Result.concat_key3("count", str(key3))] = count
            out["pval", key3] = pval
        if self.keep:
            out.update(result)
        return out