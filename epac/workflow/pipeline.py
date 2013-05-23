"""
Define "Pipeline": the primitive to build sequential execution of tasks.

@author: edouard.duchesnay@cea.fr
@author: benoit.da_mota@inria.fr
"""

## Abreviations
## tr: train
## te: test

from epac.workflow.base import BaseNode
from epac.workflow.estimators import Estimator

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
    """
    root = None
    for node in nodes:
        curr = node if isinstance(node, BaseNode) else Estimator(node)
        if not root:
            root = curr
        else:
            prev.add_child(curr)
        prev = curr
    return root