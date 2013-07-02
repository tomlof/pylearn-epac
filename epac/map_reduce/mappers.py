# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:12:28 2013

@author: jinpeng.li@cea.fr
@author: edouard.duchesnay@cea.fr
@author: benoit.da_mota@inria.fr
"""

import os
from abc import ABCMeta, abstractmethod
from epac import key_pop


def map_process(map_input, mapper):
    '''See example of epac.map_reduce.mappers.MapperSubtrees
    '''
    return mapper.map(map_input)


class Mapper(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def map(self, map_input):
        """Return results"""
        return None


class MapperSubtrees(Mapper):
    '''The mapper runs function (usually as "fit_predict") for NodesInput
    which is a dictionary of nodes.

    Example
    -------

    >>> import tempfile
    >>> import os
    >>> import numpy as np
    >>> import shutil
    >>> from functools import partial
    >>> from multiprocessing import Pool
    >>>
    >>> from epac import StoreFs
    >>> from epac.tests.wfexamples2test import WFExample2
    >>> from epac.map_reduce.inputs import NodesInput
    >>> from epac.map_reduce.split_input import SplitNodesInput
    >>> from epac.map_reduce.mappers import MapperSubtrees
    >>> from epac.map_reduce.mappers import map_process
    >>> from sklearn import datasets

    >>> ## Build dataset
    >>> ## =============
    >>> X, y = datasets.make_classification(n_samples=10,
    ...                                     n_features=20,
    ...                                     n_informative=5,
    ...                                     random_state=1)
    >>> Xy = {'X':X, 'y':y}

    >>> ## Build epac tree and save it on the disk
    >>> ## =======================================
    >>> tree_root_node = WFExample2().get_workflow()

    >>> ## Split input into several parts and create mapper
    >>> ## ================================================
    >>> num_processes = 3
    >>> node_input = NodesInput(tree_root_node.get_key())
    >>> split_node_input = SplitNodesInput(tree_root_node, num_processes=num_processes)
    >>> input_list = split_node_input.split(node_input)
    >>> mapper = MapperSubtrees(Xy=Xy,
    ...                         tree_root=tree_root_node,
    ...                         function="transform")

    >>> ## Run map processes in parallel
    >>> ## =============================
    >>> partial_map_process = partial(map_process,mapper=mapper)
    >>> pool = Pool(processes=num_processes)
    >>> res_tree_root_list = pool.map(partial_map_process, input_list)

    >>> ## Merge results from pool
    >>> for each_tree_root in res_tree_root_list:
    ...     tree_root_node.merge_tree_store(each_tree_root)

    >>> ## Run map processes in single process
    >>> ## ===================================
    >>> #for input in input_list:
    >>> #    mapper.map(input)

    >>> ## Run reduce process
    >>> ## ==================
    >>> tree_root_node.reduce()
    ResultSet(
    [{'key': SelectKBest/SVC(C=1), 'mean_score_te': 0.777777777778, 'pval_mean_score_te': 0.0, 'mean_score_tr': 0.944444444444, 'pval_mean_score_tr': 0.5},
     {'key': SelectKBest/SVC(C=3), 'mean_score_te': 0.777777777778, 'pval_mean_score_te': 0.0, 'mean_score_tr': 0.896825396825, 'pval_mean_score_tr': 0.5}])
    '''
    def __init__(self,
                 Xy,
                 tree_root,
                 store_fs=None,
                 function="transform"):
        """Initialization for the mapper

        Parameters
        ----------
        Xy:X matrix and y vertor

        tree_root: epac.BaseNode
            the root node of the epac tree

        store_fs:
            where the node want to save

        function:
            function of node
        """
        self.Xy = Xy
        self.tree_root = tree_root
        self.store_fs = store_fs
        self.function = function

    def map(self, nodes_input):
        """Run self.function for each sub_tree of map_input

        Parameters
        ----------
        nodes_input: epac.map_reduce.inputs.NodesInput

        """
        listkey = []
        for key_map_input in nodes_input:
            listkey.append(nodes_input[key_map_input])
        common_key, _ = key_pop(os.path.commonprefix(listkey))
        common_parent = None
        common_parent = self.tree_root.get_node(common_key)
        if common_parent:
            for node_root2common in common_parent.get_path_from_root():
                node_root2common = \
                    self.tree_root.get_node(node_root2common.get_key())
                # print node_root2common
                # print parent_node
                func = getattr(node_root2common, self.function)
                self.Xy = func(**self.Xy)
        # Execute what is specific to each keys
        for curr_key in listkey:
            # curr_key = listkey.__iter__().next()
            cpXy = self.Xy
            # print curr_key
            # curr_key = 'Permutations/Perm(nb=3)'
            curr_node = self.tree_root.get_node(curr_key)
            for node_common2curr in \
                    curr_node.get_path_from_node(common_parent):
                if node_common2curr is common_parent:
                    # skip commom ancestor
                    continue
                if node_common2curr is curr_node:
                    # do not process current node
                    break
                node_common2curr = \
                        self.tree_root.get_node(node_common2curr.get_key())
                func = getattr(node_common2curr, self.function)
                cpXy = func(**cpXy)
            curr_node = self.tree_root.get_node(curr_node.get_key())
            # print "Recursively run from root to current node"
            if self.store_fs:
                curr_node.store = StoreMem()
            func = getattr(curr_node, self.function)
            func(recursion=True, **cpXy)
            # print "Save results"
            if self.store_fs:
                curr_node.save_node(store=self.store_fs)
            return self.tree_root

if __name__ == "__main__":
    import doctest
    doctest.testmod()