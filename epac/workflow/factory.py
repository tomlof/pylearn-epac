# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 16:58:40 2013

@author: jinpeng.li@cea.fr

Factory class to build BaseNode
"""


from epac.workflow.wrappers import TransformNode
from epac.workflow.estimators import Estimator
from epac.workflow.base import BaseNode


class NodeFactory:

    @staticmethod
    def build(node):
        """

        Parameters
        ----------
        node: any class
            node to wraper which should implement methods
            in one of below cases:
            (-) fit and transform,
            (-) fit and predict,
            (-) transform.
        """
        if isinstance(node, BaseNode):
            return node
        elif (hasattr(node, "fit")
            and hasattr(node, "transform")
            and hasattr(node, "predict")
            ):
            return Estimator(node)
        elif (hasattr(node, "fit")
            and hasattr(node, "transform")
            ):
            return Estimator(node)
        elif (hasattr(node, "fit")
            and hasattr(node, "predict")
            ):
            return Estimator(node)
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
