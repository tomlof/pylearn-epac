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


class NodeInput(MapInput):
    ''' NodeInput for map and reduce
    This can be splited by split_input.
    '''
    def __init__(self, node):
        super(NodeInput, self).__init__()
        self[node.get_key()] = node