# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 17:26:09 2013

@author: jinpeng.li@cea.fr
@author: edouard.duchesnay@cea.fr
@author: benoit.da_mota@inria.fr
"""

from epac.map_reduce.exports import export_nodes2num_processes
from epac.map_reduce.inputs import NodesInput
import multiprocessing


class SplitInput(object):

    def __init__(self):
        self.input_list = []

    def split(self, map_input):
        self.input_list.append(map_input)
        return self.input_list


class SplitNodesInput(SplitInput):
    """ Split NodesInput into a list of epac.map_reduce.NodesInput

    Examples
    --------

    >>> from epac.tests.wfexamples2test import WFExample2
    >>> from epac.map_reduce.inputs import NodesInput
    >>> from epac.map_reduce.split_input import SplitNodesInput
    >>> tree_root_node = WFExample2().get_workflow()
    >>> nodes_input = NodesInput(tree_root_node.get_key())
    >>> split_node_input = SplitNodesInput(tree_root_node, num_processes=3)
    >>> input_list = split_node_input.split(nodes_input)
    >>> print repr(input_list)
    [{'Perms/Perm(nb=0)': 'Perms/Perm(nb=0)'}, {'Perms/Perm(nb=1)': 'Perms/Perm(nb=1)'}, {'Perms/Perm(nb=2)': 'Perms/Perm(nb=2)'}]

    """
    def __init__(self, tree_root_node, num_processes=-1):
        '''
        Parameters
        ----------
        num_processes: integer
        Building a list of num_processes node_inputs.
        if num_processes is equal to -1, num_processes is set to #CPU
        '''
        super(SplitNodesInput, self).__init__()
        self.tree_root_node = tree_root_node
        if num_processes != -1:
            self.num_processes = num_processes
        else:
            self.num_processes = multiprocessing.cpu_count()

    def split(self, nodes_input):
        '''
        Parameters
        ----------
        nodes_input: epac.map_reduce.NodesInput

        Return Value
        ------------
        list of epac.map_reduce.NodesInput in terms of self.num_processes.
        The length of this list is equal to self.num_processes
        '''
        dict_nodes_input = dict()
        for key in nodes_input:
            dict_out = export_nodes2num_processes(
                    self.tree_root_node.get_node(nodes_input[key]),
                    self.num_processes)
            ## print "dict_out=" + repr(dict_out)
            ## Convert dict_out to dict of NodesInput
            for key_dict_out in dict_out:
                list_of_node_key = dict_out[key_dict_out]
                # print "list_of_node_key=" + repr(list_of_node_key)
                for each_node_key in list_of_node_key:
                    if not (key_dict_out in dict_nodes_input.keys()):
                        dict_nodes_input[key_dict_out] = \
                                NodesInput(each_node_key)
                    else:
                        dict_nodes_input[key_dict_out].add(each_node_key)
            # print "dict_nodes_input=" + repr(dict_nodes_input)
        self.input_list = []
        for key in dict_nodes_input:
            if len(dict_nodes_input[key]) > 0:
                self.input_list.append(dict_nodes_input[key])
        return self.input_list

if __name__ == "__main__":
    import doctest
    doctest.testmod()