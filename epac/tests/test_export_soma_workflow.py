#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on 2 May 2013

@author: jinpeng.li@cea.fr
@author: edouard.duchesnay@cea.fr
@author: benoit.da_mota@inria.fr

"""

import unittest
import os
import numpy as np
import shutil

from sklearn import datasets
from epac import StoreFs

from epac.tests.wfexamples2test import get_wf_example_classes
from epac.utils import comp_2wf_reduce_res


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

    def test_soma_workflow(self):
        from soma.workflow.client import Helper
        from epac.export_multi_processes import export2somaworkflow
        list_all_examples = get_wf_example_classes()
        for example in list_all_examples:
            self._build_wdir_dataset()
            self.wf = example().get_workflow()
            self.store = StoreFs(dirpath=self.tree_root_relative_path)
            self.wf.save_tree(store=self.store)
            (wf_id, controller) = export2somaworkflow(
                in_datasets_file_relative_path=\
                self.datasets_file_relative_path,
                in_working_directory=self.my_working_directory,
                out_soma_workflow_file=self.soma_workflow_relative_path,
                in_tree_root=self.wf,
                #in_num_processes=3,
                in_is_sumbit=True,
                in_resource_id="",
                in_login="",
                in_pw="")
            Helper.wait_workflow(wf_id, controller)
            ## transfer the output files from the workflow
            Helper.transfer_output_files(wf_id, controller)
            controller.delete_workflow(wf_id)
            self._start2cmp()

    def test_multi_processes(self):
        from epac.export_multi_processes import run_multi_processes
        list_all_examples = get_wf_example_classes()
        for example in list_all_examples:
            self._build_wdir_dataset()
            self.wf = example().get_workflow()
            self.store = StoreFs(dirpath=self.tree_root_relative_path)
            self.wf.save_tree(store=self.store)
            run_multi_processes(
                in_datasets_file_relative_path=\
                    self.datasets_file_relative_path,
                in_working_directory=self.my_working_directory,
                in_tree_root=self.wf,
                in_num_processes=10,
                in_is_wait=True)
            self._start2cmp()

    def _start2cmp(self):
        os.chdir(self.my_working_directory)
        self.store = StoreFs(dirpath=self.tree_root_relative_path)
        self.swf_wf = self.store.load()
        ###################################################################
        ## Run without soma-workflow
        self.wf.fit_predict(X=self.X, y=self.y)
        self.assertTrue(comp_2wf_reduce_res(self.swf_wf, self.wf))

if __name__ == '__main__':
    unittest.main()