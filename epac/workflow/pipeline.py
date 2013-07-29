"""
Define "Pipeline": the primitive to build sequential execution of tasks.

@author: edouard.duchesnay@cea.fr
@author: benoit.da_mota@inria.fr
"""

## Abreviations
## tr: train
## te: test

from epac.workflow.base import BaseNode
from epac.workflow.factory import NodeFactory


def __insert_node_at_leaf(node, node2insert):
    """insert a node at the leaf level for a sequence

    Example
    -------
    >>> from sklearn.svm import SVC
    >>> from sklearn.lda import LDA
    >>> from sklearn.feature_selection import SelectKBest
    >>> from epac.workflow.factory import NodeFactory
    >>> from epac.workflow.pipeline import __insert_node_at_leaf
    >>>
    >>> parent_node = NodeFactory.build(SelectKBest())
    >>> child_node = NodeFactory.build(LDA())
    >>> child_of_child_node = NodeFactory.build(SVC())
    >>>
    >>> __insert_node_at_leaf(child_node, parent_node)
    >>> __insert_node_at_leaf(child_of_child_node, parent_node)
    >>>
    >>> for node in parent_node.walk_nodes():
    ...     print node
    ...
    SelectKBest
    SelectKBest/LDA
    SelectKBest/LDA/SVC

    """
    for leaf in node2insert.walk_leaves():
        leaf.add_child(node)


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
    >>> from epac import Pipe
    >>> from sklearn import datasets
    >>> from sklearn.feature_selection import SelectKBest
    >>> from sklearn.lda import LDA
    >>> from sklearn.svm import SVC
    >>>
    >>> X, y = datasets.make_classification(n_samples=12,
    ...                                     n_features=10,
    ...                                     n_informative=2,
    ...                                     random_state=1)
    >>> pipe = Pipe(SelectKBest(k=2), LDA())
    >>> pipe.transform(X=X, y=y)
    {'y': array([ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  1.,  0.,  1.]), 'X': array([[-0.34385368,  0.75623409],
           [ 0.19829972, -1.16389861],
           [-0.74715829,  0.86977629],
           [ 1.13162939,  0.90876519],
           [ 0.23009474, -0.68017257],
           [ 0.16003707, -1.55458039],
           [ 0.40349164,  1.38791468],
           [-1.11731035,  0.23476552],
           [ 1.19891788,  0.0888684 ],
           [-0.75439794, -0.90039992],
           [ 0.12015895,  2.05996541],
           [-0.20889423,  2.05313908]])}
    >>>
    >>> pipe2 = Pipe(Pipe(SelectKBest(k=2), LDA()), SVC())
    >>> pipe2.transform(X=X, y=y)
    {'y': array([ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  1.,  0.,  1.]), 'X': array([[-0.34385368,  0.75623409],
           [ 0.19829972, -1.16389861],
           [-0.74715829,  0.86977629],
           [ 1.13162939,  0.90876519],
           [ 0.23009474, -0.68017257],
           [ 0.16003707, -1.55458039],
           [ 0.40349164,  1.38791468],
           [-1.11731035,  0.23476552],
           [ 1.19891788,  0.0888684 ],
           [-0.75439794, -0.90039992],
           [ 0.12015895,  2.05996541],
           [-0.20889423,  2.05313908]])}


    """
    root = None
    prev = None
    for i in xrange(len(nodes)):
        node = nodes[i]
        curr = NodeFactory.build(node)
        if not root:
            root = curr
        else:
            __insert_node_at_leaf(curr, prev)
            # prev.add_child(curr)
        prev = curr
    return root

if __name__ == "__main__":
    import doctest
    doctest.testmod()
