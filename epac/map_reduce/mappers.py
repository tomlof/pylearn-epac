# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:12:28 2013

@author: jinpeng.li@cea.fr
@author: edouard.duchesnay@cea.fr
@author: benoit.da_mota@inria.fr
"""

import os
from abc import ABCMeta, abstractmethod
from epac import key_pop, StoreMem


class Mapper(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def map(self, data_segment):
        """Return results"""
        return None


class MapperSubtrees(Mapper):

    def __init__(self,
                 Xy,
                 tree_root,
                 store_fs,
                 function="fit_predict"):
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
        self.function = function
        self.store_fs = store_fs

    def map(self, data_segment):
        """To generate key-value-pair results from data_segment
        To generate key-value-pair results from data_segment which contains
        a list of keys.

        Parameters
        ----------
        data_segment:a list of key-node strings
        Computing the results from root to the key-node, from the key-node to
        its leaves.
        """
        listkey = data_segment
        common_key, _ = key_pop(os.path.commonprefix(listkey))
        common_parent = None
        common_parent = self.tree_root.get_node(common_key)
        if common_parent:
            for node_root2common in common_parent.get_path_from_root():
                node_root2common = \
                    self.tree_root.get_node(node_root2common.get_key())
                print node_root2common
                #print parent_node
                func = getattr(node_root2common, self.function)
                self.Xy = func(recursion=False, **self.Xy)
                print repr(self.Xy)
        # Execute what is specific to each keys
        for curr_key in listkey:
            # curr_key = listkey.__iter__().next()
            cpXy = self.Xy
            print curr_key
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
                cpXy = func(recursion=False, **cpXy)
            curr_node = self.tree_root.get_node(curr_node.get_key())
            # print "Recursively run from root to current node"
            curr_node.store = StoreMem()
            func = getattr(curr_node, self.function)
            func(recursion=True, **cpXy)
            # print "Save results"
            curr_node.save_node(store=self.store_fs)