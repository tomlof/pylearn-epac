# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 16:58:40 2013

@author: jinpeng.li@cea.fr

Factory class to build BaseNode
"""

from epac.workflow.base import BaseNode
from epac.utils import _func_get_args_names
from epac.configuration import conf
from epac.utils import _sub_dict, _as_dict
from epac.map_reduce.results import ResultSet, Result
from epac.utils import _func_get_args_names, train_test_merge, train_test_split, _dict_suffix_keys
from epac.workflow.base import key_push
from epac.workflow.estimators import InternalEstimator
from epac.workflow.estimators import LeafEstimator


class NodeFactory:

    @staticmethod
    def build(node):
        if hasattr(node, "fit") and hasattr(node, "transform"):
            return InternalEstimator(node)
        if hasattr(node, "fit") and hasattr(node, "predict"):
            return LeafEstimator(node)
        elif hasattr(node, "transform"):
            return TransformNode(node)
        else:
            raise ValueError("Fail to find a wrapper for %s. "\
                "It should implement methods in one of below cases:\n"\
                "(-) fit and transform,\n"\
                "(-) fit and predict,\n" \
                "(-) transform.\n" %
                (node.__class__.__name__))



if __name__ == "__main__":
    import doctest
    doctest.testmod()
