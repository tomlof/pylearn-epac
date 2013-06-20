# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 11:09:30 2013

@author: jinpeng.li@cea.fr
@author: edouard.duchesnay@cea.fr
@author: benoit.da_mota@inria.fr
"""

import os
import shutil
from abc import ABCMeta, abstractmethod
import multiprocessing
import tempfile
import numpy as np

from epac import StoreFs
from epac.export_multi_processes import run_multi_processes


class Engine(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit_predict(self, **Xy):
        return None


class LocalEngine(Engine):

    dataset_relative_path = "./dataset.npz"
    tree_root_relative_path = "./epac_tree"

    def __init__(self,
                 tree_root,
                 num_processes=-1):
        """Initialization for the LocalEngine
        """
        self.tree_root = tree_root
        if num_processes == -1:
            self.num_processes = multiprocessing.cpu_count()
        else:
            self.num_processes = num_processes

    def fit_predict(self, **Xy):
        tmp_work_dir_path = tempfile.mkdtemp()
        # print "tmp_work_dir_path="+tmp_work_dir_path
        np.savez(os.path.join(tmp_work_dir_path,
                 LocalEngine.dataset_relative_path), X=Xy['X'], y=Xy['y'])
        cur_work_dir = os.getcwd()
        os.chdir(tmp_work_dir_path)
        store = StoreFs(dirpath=os.path.join(
            tmp_work_dir_path,
            LocalEngine.tree_root_relative_path))
        self.tree_root.save_tree(store=store)
        run_multi_processes(
            in_datasets_file_relative_path=LocalEngine.dataset_relative_path,
            in_working_directory=tmp_work_dir_path,
            in_tree_root=self.tree_root,
            in_num_processes=self.num_processes,
            in_is_wait=True)
        self.tree_root = store.load()
        os.chdir(cur_work_dir)
        if os.path.isdir(tmp_work_dir_path):
            shutil.rmtree(tmp_work_dir_path)
        return self.tree_root


class SomaWorkflowEngine(LocalEngine):

    open_me_by_soma_workflow_gui = "open_me_by_soma_workflow_gui"

    def __init__(self,
                 tree_root,
                 num_processes=-1,
                 resource_id="",
                 login="",
                 pw=""):
        super(SomaWorkflowEngine, self).__init__(
                        tree_root=tree_root,
                        num_processes=num_processes)
        if num_processes == -1:
            self.num_processes = 20
        self.resource_id = resource_id
        self.login = login
        self.pw = pw

    def fit_predict(self, **Xy):
        from epac.export_multi_processes import export2somaworkflow
        from soma.workflow.client import Helper
        tmp_work_dir_path = tempfile.mkdtemp()
        # print "tmp_work_dir_path="+tmp_work_dir_path
        np.savez(os.path.join(tmp_work_dir_path,
                 LocalEngine.dataset_relative_path), X=Xy['X'], y=Xy['y'])
        cur_work_dir = os.getcwd()
        os.chdir(tmp_work_dir_path)
        store = StoreFs(dirpath=os.path.join(
            tmp_work_dir_path,
            LocalEngine.tree_root_relative_path))
        self.tree_root.save_tree(store=store)
        (wf_id, controller) = export2somaworkflow(
            in_datasets_file_relative_path=LocalEngine.dataset_relative_path,
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
                 LocalEngine.dataset_relative_path), X=Xy['X'], y=Xy['y'])
        cur_work_dir = os.getcwd()
        os.chdir(soma_workflow_dirpath)
        store = StoreFs(dirpath=os.path.join(
            soma_workflow_dirpath,
            LocalEngine.tree_root_relative_path))
        self.tree_root.save_tree(store=store)
        export2somaworkflow(
            in_datasets_file_relative_path=LocalEngine.dataset_relative_path,
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