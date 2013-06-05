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
from epac import Perms, CV, Pipe, Grid, CVGridSearchRefit
from epac import range_log2


def _displayres(d, indent=0):
    for key, value in d.iteritems():
        print '\t' * indent + str(key)
        if isinstance(value, dict):
            _displayres(value, indent+1)
        else:
            print '\t' * (indent+1) + str(value)


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

    def _build_first_workflow(self):
        random_state = 0
        C_values = [1, 10]
        k_values = 0
        k_max = "auto"
        n_folds_nested = 5
        n_folds = 10
        n_perms = 10
        if k_max != "auto":
            k_values = range_log2(np.minimum(int(k_max), self.n_features),
                                  add_n=True)
        else:
            k_values = range_log2(self.n_features, add_n=True)
        pipeline = CVGridSearchRefit(*[
                                     Pipe(SelectKBest(k=k),
                                     Grid(*[SVC(kernel="linear", C=C)
                                     for C in C_values]))
                                     for k in k_values],
                                     n_folds=n_folds_nested,
                                     random_state=random_state)
        self.wf = Perms(CV(pipeline, n_folds=n_folds),
                        n_perms=n_perms,
                        permute="y",
                        random_state=random_state)
        self.store = StoreFs(dirpath=self.tree_root_relative_path)
        self.wf.save_tree(store=self.store)

    def _build_second_workflow(self):
        ####################################################################
        ## EPAC WORKFLOW
        # -------------------------------------
        #             Perms                Perm (Splitter)
        #         /     |       \
        #        0      1       2            Samples (Slicer)
        #        |
        #       CV                        CV (Splitter)
        #  /       |       \
        # 0        1       2                                Folds (Slicer)
        # |        |       |
        # Pipeline     Pipeline     Pipeline                Sequence
        # |
        # 2                                   SelectKBest (Estimator)
        # |
        # Grid
        # |                     \
        # SVM(linear,C=1)   SVM(linear,C=10)  Classifiers (Estimator)
        pipeline = Pipe(SelectKBest(k=2),
                        Grid(*[SVC(kernel="linear", C=C)
                        for C in [1, 3]]))
        self.wf = Perms(CV(pipeline, n_folds=3),
                        n_perms=3,
                        permute="y",
                        y=self.y)
        self.store = StoreFs(dirpath=self.tree_root_relative_path)
        self.wf.save_tree(store=self.store)

    def _example_one(self):
        self._build_wdir_dataset()
        self._build_first_workflow()

    def _example_two(self):
        self._build_wdir_dataset()
        self._build_second_workflow()

#    def test_example_one(self):
#        self._build_wdir_dataset()
#        self._build_second_workflow()
#        self.wf.fit_predict(X=self.X, y=self.y)
#        self.wf.reduce()
#        #print self.wf.reduce()

#    def test_example_two(self):
#        self._build_wdir_dataset()
#        self._build_second_workflow()
#        self.wf.fit_predict(X=self.X, y=self.y)
#        self.wf.reduce()
#        #print self.wf.reduce()

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_soma_workflow_cluster(self):
        from soma.workflow.client import Helper
        from epac.export_multi_processes import export2somaworkflow
#        self._example_one()
#        (wf_id, controller) = export2somaworkflow(
#            in_datasets_file_relative_path=self.datasets_file_relative_path,
#            in_working_directory=self.my_working_directory,
#            out_soma_workflow_file=self.soma_workflow_relative_path,
#            in_tree_root=self.wf,
#            in_is_sumbit=True,
#            in_resource_id="ed203246@gabriel",
#            in_login="ed203246",
#            in_pw="")
#        ## wait the workflow to finish
#        Helper.wait_workflow(wf_id, controller)
#        ## transfer the output files from the workflow
#        Helper.transfer_output_files(wf_id, controller)
#        controller.delete_workflow(wf_id)
#        self._start2cmp()
#        self._example_two()
#        (wf_id, controller) = export2somaworkflow(
#            in_datasets_file_relative_path=self.datasets_file_relative_path,
#            in_working_directory=self.my_working_directory,
#            out_soma_workflow_file=self.soma_workflow_relative_path,
#            in_tree_root=self.wf,
#            in_is_sumbit=True,
#            in_resource_id="ed203246@gabriel",
#            in_login="ed203246",
#            in_pw="")
#        ## wait the workflow to finish
#        Helper.wait_workflow(wf_id, controller)
#        ## transfer the output files from the workflow
#        Helper.transfer_output_files(wf_id, controller)
#        controller.delete_workflow(wf_id)
#        self._start2cmp()

    def test_soma_workflow(self):
        from soma.workflow.client import Helper
        from epac.export_multi_processes import export2somaworkflow
#        self._example_one()
#        (wf_id, controller) = export2somaworkflow(
#            in_datasets_file_relative_path=self.datasets_file_relative_path,
#            in_working_directory=self.my_working_directory,
#            out_soma_workflow_file=self.soma_workflow_relative_path,
#            in_tree_root=self.wf,
#            #in_num_processes=3,
#            in_is_sumbit=True,
#            in_resource_id="",
#            in_login="",
#            in_pw=""
#        )
#        ## wait the workflow to finish
#        Helper.wait_workflow(wf_id, controller)
#        ## transfer the output files from the workflow
#        Helper.transfer_output_files(wf_id, controller)
#        #controller.delete_workflow(wf_id)
#        self._start2cmp()
        self._example_two()
        (wf_id, controller) = export2somaworkflow(
            in_datasets_file_relative_path=self.datasets_file_relative_path,
            in_working_directory=self.my_working_directory,
            out_soma_workflow_file=self.soma_workflow_relative_path,
            in_tree_root=self.wf,
            #in_num_processes=3,
            in_is_sumbit=True,
            in_resource_id="",
            in_login="",
            in_pw=""
        )
        ## wait the workflow to finish
        Helper.wait_workflow(wf_id, controller)
        ## transfer the output files from the workflow
        Helper.transfer_output_files(wf_id, controller)
        #controller.delete_workflow(wf_id)
        self._start2cmp()
        

#    def test_soma_workflow_nodes(self):
#        from soma.workflow.client import Helper
#        from epac.export_multi_processes import export2somaworkflow
#        self._example_one()
#        nodes = self.wf.get_node(regexp="*/Perms/*")
#        print "nodes="+repr(nodes)
#        (wf_id, controller) = export2somaworkflow(
#            in_datasets_file_relative_path=self.datasets_file_relative_path,
#            in_working_directory=self.my_working_directory,
#            out_soma_workflow_file=self.soma_workflow_relative_path,
#            in_nodes=nodes,
#            in_is_sumbit=True,
#            in_resource_id="",
#            in_login="",
#            in_pw=""
#        )
#        ## wait the workflow to finish
#        Helper.wait_workflow(wf_id, controller)
#        ## transfer the output files from the workflow
#        Helper.transfer_output_files(wf_id, controller)
#        controller.delete_workflow(wf_id)
#        self._start2cmp()

#    def test_multi_processes_n_processes(self):
#        for n_processes in range(2, 5):
#            print "n_processes=" + repr(n_processes)
#            from epac.export_multi_processes import run_multi_processes
#            print "_example_one"
#            self._example_one()
#            run_multi_processes(
#                in_datasets_file_relative_path=
#                self.datasets_file_relative_path,
#                in_working_directory=self.my_working_directory,
#                in_tree_root=self.wf,
#                in_num_processes=n_processes,
#                in_is_wait=True)
#            self._start2cmp()
#            print "_example_two"
#            self._example_two()
#            run_multi_processes(
#                in_datasets_file_relative_path=
#                self.datasets_file_relative_path,
#                in_working_directory=self.my_working_directory,
#                in_tree_root=self.wf,
#                in_num_processes=n_processes,
#                in_is_wait=True)
#            self._start2cmp()

#    def test_multi_processes(self):
#        from epac.export_multi_processes import run_multi_processes
#        self._example_one()
#        run_multi_processes(
#            in_datasets_file_relative_path=self.datasets_file_relative_path,
#            in_working_directory=self.my_working_directory,
#            in_tree_root=self.wf,
#            in_num_processes=3,
#            in_is_wait=True)
#        self._start2cmp()
#        self._example_two()
#        run_multi_processes(
#            in_datasets_file_relative_path=self.datasets_file_relative_path,
#            in_working_directory=self.my_working_directory,
#            in_tree_root=self.wf,
#            in_num_processes=3,
#            in_is_wait=True)
#        self._start2cmp()

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
        _displayres(R1)
        _displayres(R2)
        self.assertTrue(_isequal(R1, R2))
        return

if __name__ == '__main__':
    unittest.main()
