# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 16:58:40 2013

@author: jinpeng.li@cea.fr

Factory class to build BaseNode
"""


from epac.workflow.wrappers import TransformNode
from epac.workflow.estimators import InternalEstimator
from epac.workflow.estimators import LeafEstimator


class NodeFactory:

    @staticmethod
    def build(node, is_leaf=True):
        """

        Parameters
        ----------
        node: any class
            node to wraper which should implement methods
            in one of below cases:

        is_leaf: boolean
            is_leaf is an option only for the classifier which implement all
            three methods, fit, transform and predict. Because we don't know
            if it is leaf, we need this parameter, is_leaf.
            For example, PCA classifier in scikit-learn contains three methods.
            It coule either non-leaf node or leaf node.

        """
        if (hasattr(node, "fit")
            and hasattr(node, "transform")
            and hasattr(node, "predict")
            ):
            # For example, PCA classifier contains three methods
            # So we need to know if it is a leaf.
            if is_leaf:
                return LeafEstimator(node)
            else:
                return InternalEstimator(node)
        elif (hasattr(node, "fit")
            and hasattr(node, "transform")
            ):
            return InternalEstimator(node)
        elif (hasattr(node, "fit")
            and hasattr(node, "predict")
            ):
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