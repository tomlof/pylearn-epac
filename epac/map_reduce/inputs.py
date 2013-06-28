# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:27:09 2013

@author: jinpeng.li@cea.fr
@author: edouard.duchesnay@cea.fr
@author: benoit.da_mota@inria.fr
"""


class Input(dict):
    pass


class MapInput(Input):
    pass


class ReduceInput(Input):
    pass


class NodesInput(MapInput):
    ''' NodesInput for map
    This can be splited by split_input. All key strings have been saved.

    Example
    -------
    >>> from epac.map_reduce.inputs import NodesInput
    >>> nodes_input = NodesInput('Perms/Perm(nb=0)')
    >>> nodes_input.add('Perms/Perm(nb=1)')
    >>> nodes_input.add('Perms/Perm(nb=2)')
    >>> print repr(nodes_input)
    {'Perms/Perm(nb=2)': 'Perms/Perm(nb=2)', 'Perms/Perm(nb=1)': 'Perms/Perm(nb=1)', 'Perms/Perm(nb=0)': 'Perms/Perm(nb=0)'}
    '''
    def __init__(self, node_key):
        super(NodesInput, self).__init__()
        self.add(node_key)

    def add(self, node_key):
        '''
        Parameter
        ---------
        node_key is a string. We don't save it save a BaseNode
        since VirtualList cannot be permanent.
        '''
        self[node_key] = node_key

if __name__ == "__main__":
    import doctest
    doctest.testmod()