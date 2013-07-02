# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 11:09:30 2013

@author: jinpeng.li@cea.fr
@author: edouard.duchesnay@cea.fr
@author: benoit.da_mota@inria.fr
"""

import os
import shutil
import multiprocessing
import tempfile
import numpy as np

from abc import ABCMeta, abstractmethod

from epac import StoreFs


class Engine(object):
    __metaclass__ = ABCMeta

    def __init__(self, split_input, mapper, reducer):
        self.split_input = split_input
        self.mapper = mapper
        self.reducer = reducer

    @abstractmethod
    def run(self, **Xy):
        pass
#        list_map_input = self.split_input(map_input)
#        list_reduce_input = []
#        for each_map_input in list_map_input:
#            reduce_input = self.mapper.map(each_map_input)
#            list_reduce_input.append(reduce_input)
#        ##the shuffle step have been skiped since we dont have this step
#        self.reducer.reduce(list_reduce_input)

class LocalEngine(Engine):
    '''LocalEninge run a specified function for a epac tree in parallel

    Example
    -------

from sklearn import datasets

from epac.map_reduce.engine import LocalEngine
from epac.tests.wfexamples2test import WFExample2

## Build dataset
## =============
X, y = datasets.make_classification(n_samples=10,
                                    n_features=20,
                                    n_informative=5,
                                    random_state=1)
Xy = {'X':X, 'y':y}

## Build epac tree
## ===============
tree_root_node = WFExample2().get_workflow()

## Build LocalEngine
## =================
local_engine = LocalEngine(tree_root_node,
                           function_name="transform",
                           num_processes=3)
tree_root_node = local_engine.run(**Xy)

## Run reduce process
## ==================
## pval_mean_score_te might be different since permutation is random
tree_root_node.reduce()
    '''
    tree_root_relative_path = "./epac_tree"

    def __init__(self,
                 tree_root,
                 function_name="transform",
                 num_processes=-1):
        """Initialization for the LocalEngine

        Parameters
        ----------
        tree_root: BaseNode

        function_name: string
            The name of function need to be executed through all nodes in
            epac tree

        num_processes: integer
            Run map process in #processes
        """
        self.tree_root = tree_root
        self.function_name = function_name
        if num_processes == -1:
            self.num_processes = multiprocessing.cpu_count()
        else:
            self.num_processes = num_processes

    def run(self, **Xy):
        from functools import partial
        from multiprocessing import Pool
        from epac.map_reduce.inputs import NodesInput
        from epac.map_reduce.split_input import SplitNodesInput
        from epac.map_reduce.mappers import MapperSubtrees
        from epac.map_reduce.mappers import map_process

        tmp_work_dir_path = tempfile.mkdtemp()
        store = StoreFs(dirpath=os.path.join(
            tmp_work_dir_path,
            LocalEngine.tree_root_relative_path))
        self.tree_root.save_tree(store=store)

        ## Split input into several parts and create mapper
        ## ================================================
        node_input = NodesInput(self.tree_root.get_key())
        split_node_input = SplitNodesInput(self.tree_root,
                                           num_processes=self.num_processes)
        input_list = split_node_input.split(node_input)
        mapper = MapperSubtrees(Xy,
                                self.tree_root, store,
                                self.function_name)
        ## Run map processes in parallel
        ## =============================
        partial_map_process = partial(map_process, mapper=mapper)
        pool = Pool(processes=self.num_processes)
        pool.map(partial_map_process, input_list)

        ## Load results tree and remove temp working directory
        ## ===================================================
        self.tree_root = store.load()
        if os.path.isdir(tmp_work_dir_path):
            shutil.rmtree(tmp_work_dir_path)
        return self.tree_root


class SomaWorkflowEngine(LocalEngine):
    '''Using soma-workflow to run epac tree in parallel

    Example
    -------

from sklearn import datasets
from epac.map_reduce.engine import SomaWorkflowEngine
from epac.tests.wfexamples2test import WFExample2

## Build dataset
## =============
X, y = datasets.make_classification(n_samples=10,
                                    n_features=20,
                                    n_informative=5,
                                    random_state=1)
Xy = {'X':X, 'y':y}
## Build epac tree
## ===============
tree_root_node = WFExample2().get_workflow()

## Build SomaWorkflowEngine
## =================
sfw_engine = SomaWorkflowEngine(tree_root=tree_root_node,
                                function_name="transform",
                                num_processes=3)
tree_root_node = sfw_engine.run(**Xy)

## Run reduce process
## ==================
tree_root_node.reduce()

    '''
    dataset_relative_path = "./dataset.npz"
    open_me_by_soma_workflow_gui = "open_me_by_soma_workflow_gui"

    def __init__(self,
                 tree_root,
                 function_name="transform",
                 num_processes=-1,
                 resource_id="",
                 login="",
                 pw=""):
        super(SomaWorkflowEngine, self).__init__(
                        tree_root=tree_root,
                        function_name=function_name,
                        num_processes=num_processes)
        if num_processes == -1:
            self.num_processes = 20
        self.resource_id = resource_id
        self.login = login
        self.pw = pw

    def run(self, **Xy):
        from epac.map_reduce.exports import export2somaworkflow
        from soma.workflow.client import Helper
        tmp_work_dir_path = tempfile.mkdtemp()
        # print "tmp_work_dir_path="+tmp_work_dir_path
        np.savez(os.path.join(tmp_work_dir_path,
                 SomaWorkflowEngine.dataset_relative_path), **Xy)
        cur_work_dir = os.getcwd()
        os.chdir(tmp_work_dir_path)
        store = StoreFs(dirpath=os.path.join(
            tmp_work_dir_path,
            SomaWorkflowEngine.tree_root_relative_path))
        self.tree_root.save_tree(store=store)
        (wf_id, controller) = export2somaworkflow(
            in_datasets_file_relative_path=SomaWorkflowEngine.dataset_relative_path,
            in_working_directory=tmp_work_dir_path,
            out_soma_workflow_file=
            SomaWorkflowEngine.open_me_by_soma_workflow_gui,
            in_num_processes=self.num_processes,
            in_tree_root=self.tree_root,
            in_is_sumbit=True,
            in_resource_id=self.resource_id,
            in_login=self.login,
            in_pw=self.pw)
        Helper.wait_workflow(wf_id, controller)
        Helper.transfer_output_files(wf_id, controller)
        controller.delete_workflow(wf_id)
        self.tree_root = store.load()
        os.chdir(cur_work_dir)
        if os.path.isdir(tmp_work_dir_path):
            shutil.rmtree(tmp_work_dir_path)
        return self.tree_root

    def export_to_gui(self, soma_workflow_dirpath, **Xy):
        from epac.export_multi_processes import export2somaworkflow
        if(os.path.isdir(soma_workflow_dirpath)):
            raise ValueError('%s is not an empty directory.' %
                (soma_workflow_dirpath))
        os.mkdir(soma_workflow_dirpath)
        np.savez(os.path.join(soma_workflow_dirpath,
                 SomaWorkflowEngine.dataset_relative_path), X=Xy['X'], y=Xy['y'])
        cur_work_dir = os.getcwd()
        os.chdir(soma_workflow_dirpath)
        store = StoreFs(dirpath=os.path.join(
            soma_workflow_dirpath,
            SomaWorkflowEngine.tree_root_relative_path))
        self.tree_root.save_tree(store=store)
        export2somaworkflow(
            in_datasets_file_relative_path=SomaWorkflowEngine.dataset_relative_path,
            in_working_directory=soma_workflow_dirpath,
            out_soma_workflow_file=
                SomaWorkflowEngine.open_me_by_soma_workflow_gui,
            in_tree_root=self.tree_root,
            in_num_processes=self.num_processes)
        os.chdir(cur_work_dir)

    @staticmethod
    def load_from_gui(soma_workflow_dirpath):
        store = StoreFs(dirpath=os.path.join(
            soma_workflow_dirpath,
            LocalEngine.tree_root_relative_path))
        tree_root = store.load()
        return tree_root

if __name__ == "__main__":
    import doctest
    doctest.testmod()
