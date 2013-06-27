# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 17:26:09 2013

@author: jinpeng.li@cea.fr
@author: edouard.duchesnay@cea.fr
@author: benoit.da_mota@inria.fr
"""

from epac.map_reduce.exports import export_nodes2num_processes
import multiprocessing


class SplitInput(object):

    def __init__(self):
        self.input_list = []

    def split(self, map_input):
        self.input_list.append(map_input)
        return self.input_list


class SplitNodeInput(object):
    """ Split NodeInput into a list of NodeInputs

    Examples
    -------

    >>> from epac.tests.wfexamples2test import WFExample1
    >>> from epac.map_reduce.inputs import NodeInput
    >>> from epac.map_reduce.split_input import SplitNodeInput
    >>> tree_root_node = WFExample1().get_workflow()
    >>> node_input = NodeInput(tree_root_node)
    >>> split_node_input = SplitNodeInput(num_processes=3)
    >>> input_list = split_node_input.split(node_input)
    >>> print repr(input_list)
    [['Methods/SVC(C=1)'], ['Methods/SVC(C=3)'], []]
    """
    def __init__(self, num_processes=-1):
        '''
        Parameters
        ----------
        num_processes: integer
        Building a list of num_processes node_inputs.
        if num_processes is equal to -1, num_processes is set to #CPU
        '''
        super(SplitNodeInput, self).__init__()
        if num_processes != -1:
            self.num_processes = num_processes
        else:
            self.num_processes = multiprocessing.cpu_count()

    def split(self, node_input):
        '''
        Parameters
        ----------
        node_input: epac.map_reduce.NodeInput
        '''
        jobs = dict()
        for key in node_input:
            dict_out = export_nodes2num_processes(node_input[key],
                                                  self.num_processes)
            for key_dict_out in dict_out:
                # print "key_dict_out=" + repr(key_dict_out)
                if key_dict_out in jobs.keys():
                    jobs[key_dict_out] = jobs[key_dict_out] + \
                                            dict_out[key_dict_out]
                else:
                    jobs[key_dict_out] = dict_out[key_dict_out]
        # print repr(jobs)
        self.input_list = []
        for key_job in jobs:
            # print repr(jobs[key_job])
            self.input_list.append(jobs[key_job])
        return self.input_list

if __name__ == "__main__":
    import doctest
    doctest.testmod()