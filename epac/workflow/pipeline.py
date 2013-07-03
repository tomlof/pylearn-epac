"""
Define "Pipeline": the primitive to build sequential execution of tasks.

@author: edouard.duchesnay@cea.fr
@author: benoit.da_mota@inria.fr
"""

## Abreviations
## tr: train
## te: test

from epac.workflow.base import BaseNode
from epac.workflow.estimators import LeafEstimator, InternalEstimator

## ======================================================================== ##
## ==                                                                    == ##
## == Pipelineuential nodes
## ==
## ======================================================================== ##

def Pipe(*nodes):
    """
    Pipelineuential execution of Nodes.

    Parameters
    ----------
    task [, task]*

    Example
    -------
    >>> from sklearn.svm import SVC
    >>> from sklearn.feature_selection import SelectKBest
    >>> from epac import Pipe
    >>> pipe = Pipe(SelectKBest(k=2), SVC())
    """
    root = None
    
    for i in xrange(len(nodes)):
        node = nodes[i]
        if  not isinstance(node, BaseNode):
            if i == len(nodes) - 1:    
                curr = LeafEstimator(node)
            else:
                curr = InternalEstimator(node)
        if not root:
            root = curr
        else:
            prev.add_child(curr)
        prev = curr
    return root