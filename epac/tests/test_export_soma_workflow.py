#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on 2 May 2013

@author: edouard.duchesnay@cea.fr
@author: benoit.da_mota@inria.fr
@author: jinpeng.li@cea.fr

"""

import unittest
import os
import numpy as np
import shutil

from sklearn import datasets
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest

from epac import StoreFs
from epac import conf
from epac import CVBestSearchRefit, Pipe, CV, Perms, Methods
from epac import range_log2

from epac.tests.wfexamples2test import get_wf_example_classes


def _displayres(d, indent=0):
    print repr(d)
#    for key, value in d.iteritems():
#        print '\t' * indent + str(key)
#        if isinstance(value, dict):
#            _displayres(value, indent + 1)
#        else:
#            print '\t' * (indent + 1) + str(value)


def _is_numeric_paranoid(obj):
    return isinstance(obj, (int, long, float, complex))


def _is_dict_or_array_or_list(obj):
    if type(obj) is np.ndarray:
        return True
    if type(obj) is list:
        return True
    if type(obj) is dict:
        return True
    return False


def _is_array_or_list(obj):
    if type(obj) is np.ndarray:
        return True
    if type(obj) is list:
        return True
    return False


def _isequal(obj1, obj2):
    _EPSILON = 0.00001
    if _is_numeric_paranoid(obj1):
        if (np.absolute(obj1 - obj2) > _EPSILON):
            return False
        else:
            return True
    elif (isinstance(obj1, dict)):
        for key in obj1.keys():
            if not _isequal(obj1[key], obj2[key]):
                return False
        return True
    elif (_is_array_or_list(obj1)):
        obj1 = np.asarray(list(obj1))
        obj2 = np.asarray(list(obj2))
        for index in xrange(len(obj1.flat)):
            if not _isequal(obj1.flat[index], obj2.flat[index]):
                return False
        return True
    else:
        return obj1 == obj2


class EpacWorkflowTest(unittest.TestCase):
    X = None
    y = None
    n_samples = 100
    n_features = int(1E03)
    ###################################################################
    ###+my_epac_working_directory
    ###  -epac_datasets.npz
    ###  +storekeys
    ###    +...
    ###  -epac_workflow_example
    ## Setup a working directory (my_working_directory)
    my_working_directory = "/tmp/my_working_directory"
    ## key_file and datasets_file should be ***RELATIVE*** path
    ## It is mandatory for mapping path in soma-workflow
    ## since my_working_directory will be changed on the cluster
    datasets_file_relative_path = "./epac_datasets.npz"
    tree_root_relative_path = "./epac_tree"
    soma_workflow_relative_path = "./epac_workflow_example"
    ## the root of EPAC tree
    wf = None
    store = None

    def _build_wdir_dataset(self):
        ###################################################################
        ## Clean and change the working directory
        # so that we can use relative path in the directory
        # my_working_directory
        if os.path.isdir(self.my_working_directory):
            shutil.rmtree(self.my_working_directory)
        os.mkdir(self.my_working_directory)
        os.chdir(self.my_working_directory)
        ####################################################################
        ## DATASET
        self.X, self.y = datasets.make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=5)
        np.savez(self.datasets_file_relative_path, X=self.X, y=self.y)



    def setUp(self):
        pass

    def tearDown(self):
        pass

#    def test_soma_workflow_cluster(self):
#        from soma.workflow.client import Helper
#        from epac.export_multi_processes import export2somaworkflow
#        list_all_examples = get_wf_example_classes()
#        for example in list_all_examples:
#            self._build_wdir_dataset()
#            self.wf = example().get_workflow()
#            self.store = StoreFs(dirpath=self.tree_root_relative_path)
#            self.wf.save_tree(store=self.store)
#            (wf_id, controller) = export2somaworkflow(
#                in_datasets_file_relative_path=self.datasets_file_relative_path,
#                in_working_directory=self.my_working_directory,
#                out_soma_workflow_file=self.soma_workflow_relative_path,
#                in_tree_root=self.wf,
#                in_is_sumbit=True,
#                in_resource_id="ed203246@gabriel",
#                in_login="ed203246",
#                in_pw="")
#            Helper.wait_workflow(wf_id, controller)
#            ## transfer the output files from the workflow
#            Helper.transfer_output_files(wf_id, controller)
#            controller.delete_workflow(wf_id)
#            self._start2cmp()
#
#    def test_soma_workflow(self):
#        from soma.workflow.client import Helper
#        from epac.export_multi_processes import export2somaworkflow
#        list_all_examples = get_wf_example_classes()
#        for example in list_all_examples:
#            self._build_wdir_dataset()
#            self.wf = example().get_workflow()
#            self.store = StoreFs(dirpath=self.tree_root_relative_path)
#            self.wf.save_tree(store=self.store)
#            (wf_id, controller) = export2somaworkflow(
#                in_datasets_file_relative_path=\
#                self.datasets_file_relative_path,
#                in_working_directory=self.my_working_directory,
#                out_soma_workflow_file=self.soma_workflow_relative_path,
#                in_tree_root=self.wf,
#                #in_num_processes=3,
#                in_is_sumbit=True,
#                in_resource_id="",
#                in_login="",
#                in_pw="")
#            Helper.wait_workflow(wf_id, controller)
#            ## transfer the output files from the workflow
#            Helper.transfer_output_files(wf_id, controller)
#            controller.delete_workflow(wf_id)
#            self._start2cmp()

    def test_multi_processes(self):
        from epac.export_multi_processes import run_multi_processes
        list_all_examples = get_wf_example_classes()
        for example in list_all_examples:
            self._build_wdir_dataset()
            self.wf = example().get_workflow()
            self.store = StoreFs(dirpath=self.tree_root_relative_path)
            self.wf.save_tree(store=self.store)
            run_multi_processes(
            in_datasets_file_relative_path=self.datasets_file_relative_path,
            in_working_directory=self.my_working_directory,
            in_tree_root=self.wf,
            in_num_processes=10,
            in_is_wait=True)
            self._start2cmp()

    def _start2cmp(self):
        os.chdir(self.my_working_directory)
        self.store = StoreFs(dirpath=self.tree_root_relative_path)
        self.swf_wf = self.store.load()
        self.res_swf = self.swf_wf.reduce()  # Reduce process
        ###################################################################
        ## Run without soma-workflow
        self.wf.fit_predict(X=self.X, y=self.y)
        self.res_epac = self.wf.reduce()
        self._compare_res(self.res_epac, self.res_swf)

    def _compare_res(self, R1, R2):
#        _displayres(R1)
#        _displayres(R2)
        self.assertTrue(_isequal(R1, R2))
        return

if __name__ == '__main__':
    unittest.main()
