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
import sys
import socket

from abc import ABCMeta, abstractmethod

from epac import StoreFs
from epac.errors import NoSomaWFError, NoEpacTreeRootError
from epac.configuration import conf
from epac.map_reduce.split_input import SplitNodesInput
from epac.map_reduce.inputs import NodesInput


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
    '''LocalEninge run function for each node of epac tree in parallel

    Example
    -------

    >>> from sklearn import datasets
    >>> from epac.map_reduce.engine import LocalEngine
    >>> from epac.tests.wfexamples2test import WFExample2

    >>> ## Build dataset
    >>> ## =============
    >>> X, y = datasets.make_classification(n_samples=10,
    ...                                     n_features=20,
    ...                                     n_informative=5,
    ...                                     random_state=1)
    >>> Xy = {'X':X, 'y':y}

    >>> ## Build epac tree
    >>> ## ===============
    >>> tree_root_node = WFExample2().get_workflow()

    >>> ## Build LocalEngine
    >>> ## =================
    >>> local_engine = LocalEngine(tree_root_node,
    ...                            function_name="transform",
    ...                            num_processes=3)
    >>> tree_root_node = local_engine.run(**Xy)

    >>> ## Run reduce process
    >>> ## ==================
    >>> tree_root_node.reduce()
    ResultSet(
    [{'key': SelectKBest/SVC(C=1), 'y/test/score_recall_mean/pval': [ 0.], 'y/test/score_recall/pval': [ 0.  0.], 'y/test/score_accuray': 0.8, 'y/test/score_f1/pval': [ 0.  0.], 'y/test/score_precision/pval': [ 0.  0.], 'y/test/score_precision': [ 0.8  0.8], 'y/test/score_recall': [ 0.8  0.8], 'y/test/score_f1': [ 0.8  0.8], 'y/test/score_recall_mean': 0.8, 'y/test/score_accuray/pval': [ 0.]},
     {'key': SelectKBest/SVC(C=3), 'y/test/score_recall_mean/pval': [ 0.], 'y/test/score_recall/pval': [ 0.  0.], 'y/test/score_accuray': 0.8, 'y/test/score_f1/pval': [ 0.  0.], 'y/test/score_precision/pval': [ 0.  0.], 'y/test/score_precision': [ 0.8  0.8], 'y/test/score_recall': [ 0.8  0.8], 'y/test/score_f1': [ 0.8  0.8], 'y/test/score_recall_mean': 0.8, 'y/test/score_accuray/pval': [ 0.]}])

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
        if num_processes == 0:
            num_processes = 1
        if num_processes < 0:
            self.num_processes = multiprocessing.cpu_count()
        else:
            self.num_processes = num_processes

    def run(self, **Xy):
        from functools import partial
        from multiprocessing import Pool

        from epac.map_reduce.mappers import MapperSubtrees
        from epac.map_reduce.mappers import map_process

        ## Split input into several parts and create mapper
        ## ================================================
        node_input = NodesInput(self.tree_root.get_key())
        split_node_input = SplitNodesInput(self.tree_root,
                                           num_processes=self.num_processes)
        input_list = split_node_input.split(node_input)
        mapper = MapperSubtrees(Xy=Xy,
                                tree_root=self.tree_root,
                                function=self.function_name)
        ## Run map processes in parallel
        ## =============================
        partial_map_process = partial(map_process, mapper=mapper)
        pool = Pool(processes=len(input_list))
        res_tree_root_list = pool.map(partial_map_process, input_list)
        for each_tree_root in res_tree_root_list:
            self.tree_root.merge_tree_store(each_tree_root)
        return self.tree_root


class SomaWorkflowEngine(LocalEngine):
    '''Using soma-workflow to run epac tree in parallel
    '''
    dataset_relative_path = "./dataset.npz"
    open_me_by_soma_workflow_gui = "open_me_by_soma_workflow_gui"

    def __init__(self,
                 tree_root,
                 function_name="transform",
                 num_processes=-1,
                 resource_id="",
                 login="",
                 pw="",
                 remove_finished_wf=True,
                 remove_local_tree=True):
        super(SomaWorkflowEngine, self).__init__(
                        tree_root=tree_root,
                        function_name=function_name,
                        num_processes=num_processes)
        if num_processes == -1:
            self.num_processes = 20
        self.resource_id = resource_id
        self.login = login
        self.pw = pw
        self.remove_finished_wf = remove_finished_wf
        self.remove_local_tree  = remove_local_tree

    def _save_job_list(self,
                        working_directory,
                        nodesinput_list):
        '''Write job list into working_directory as 0.job, 1.job, etc.

        Parameters
        ----------
        working_directory: string
            directory to write job list

        nodesinput_list: list of NodesInput
            This is for parallel computing for each element in the list.
            All of them are saved separately in working_directory.
            
        Example
        -------
        >>> from epac.map_reduce.engine import SomaWorkflowEngine
        >>> nodesinput_list = [{'Perms/Perm(nb=0)': 'Perms/Perm(nb=0)'}, 
        ...                    {'Perms/Perm(nb=1)': 'Perms/Perm(nb=1)'}, 
        ...                    {'Perms/Perm(nb=2)': 'Perms/Perm(nb=2)'}]
        >>> working_directory =  "/tmp"
        >>> swf_engine = SomaWorkflowEngine(None)
        >>> swf_engine._save_job_list(working_directory, nodesinput_list)
        ['./0.job', './1.job', './2.job']
        '''
        keysfile_list = list()
        jobi = 0
        for nodesinput in nodesinput_list:
            keysfile = "."+os.path.sep+repr(jobi)+"."+conf.SUFFIX_JOB
            keysfile_list.append(keysfile)
            # print "in_working_directory="+in_working_directory
            # print "keysfile="+keysfile
            abs_keysfile = os.path.join(working_directory, keysfile)
            f = open(abs_keysfile, 'w')
            for key_signature in nodesinput:
                f.write("%s\n" % key_signature)
            f.close()
            jobi = jobi + 1
        return keysfile_list

    def run(self, **Xy):
        '''Run soma-workflow without gui

        Example
        -------

        >>> from sklearn import datasets
        >>> from epac.map_reduce.engine import SomaWorkflowEngine
        >>> from epac.tests.wfexamples2test import WFExample2

        >>> ## Build dataset
        >>> ## =============
        >>> X, y = datasets.make_classification(n_samples=10,
        ...                                     n_features=20,
        ...                                     n_informative=5,
        ...                                     random_state=1)
        >>> Xy = {'X':X, 'y':y}

        >>> ## Build epac tree
        >>> ## ===============
        >>> tree_root_node = WFExample2().get_workflow()

        >>> ## Build SomaWorkflowEngine and run function for each node
        >>> ## =======================================================
        >>> sfw_engine = SomaWorkflowEngine(tree_root=tree_root_node,
        ...                                 function_name="trasform",
        ...                                 num_processes=3)
        >>> tree_root_node = sfw_engine.run(**Xy)

        >>> ## Run reduce process
        >>> ## ==================
        >>> tree_root_node.reduce()
        ResultSet(
        [{'key': SelectKBest/SVC(C=1), 'y/test/score_recall_mean/pval': [ 0.], 'y/test/score_recall/pval': [ 0.  0.], 'y/test/score_accuray': 0.8, 'y/test/score_f1/pval': [ 0.  0.], 'y/test/score_precision/pval': [ 0.  0.], 'y/test/score_precision': [ 0.8  0.8], 'y/test/score_recall': [ 0.8  0.8], 'y/test/score_f1': [ 0.8  0.8], 'y/test/score_recall_mean': 0.8, 'y/test/score_accuray/pval': [ 0.]},
         {'key': SelectKBest/SVC(C=3), 'y/test/score_recall_mean/pval': [ 0.], 'y/test/score_recall/pval': [ 0.  0.], 'y/test/score_accuray': 0.8, 'y/test/score_f1/pval': [ 0.  0.], 'y/test/score_precision/pval': [ 0.  0.], 'y/test/score_precision': [ 0.8  0.8], 'y/test/score_recall': [ 0.8  0.8], 'y/test/score_f1': [ 0.8  0.8], 'y/test/score_recall_mean': 0.8, 'y/test/score_accuray/pval': [ 0.]}])
        '''
        try:
            from soma.workflow.client import Job, Workflow
            from soma.workflow.client import Helper, FileTransfer
            from soma.workflow.client import WorkflowController
        except ImportError:
            errmsg = "No soma-workflow is found. "\
                "Please verify your soma-worklow"\
                "on your computer (e.g. PYTHONPATH) \n"
            sys.stderr.write(errmsg)
            sys.stdout.write(errmsg)
            raise NoSomaWFError
        tmp_work_dir_path = tempfile.mkdtemp()
        cur_work_dir = os.getcwd()
        os.chdir(tmp_work_dir_path)
        ft_working_directory = FileTransfer(is_input=True,
                                        client_path=tmp_work_dir_path,
                                        name="working directory")
        ## Save the database and tree to working directory
        ## ===============================================
        np.savez(os.path.join(tmp_work_dir_path,
                 SomaWorkflowEngine.dataset_relative_path), **Xy)
        store = StoreFs(dirpath=os.path.join(
            tmp_work_dir_path,
            SomaWorkflowEngine.tree_root_relative_path))
        self.tree_root.save_tree(store=store)

        ## Subtree job allocation on disk
        ## ==============================
        node_input = NodesInput(self.tree_root.get_key())
        split_node_input = SplitNodesInput(self.tree_root,
                                           num_processes=self.num_processes)
        nodesinput_list = split_node_input.split(node_input)
        keysfile_list = self._save_job_list(tmp_work_dir_path,
                                            nodesinput_list)
        ## Build soma-workflow
        ## ===================
        jobs = [Job(command=[u"epac_mapper",
                         u'--datasets', '"%s"' %
                         (SomaWorkflowEngine.dataset_relative_path),
                         u'--keysfile', '"%s"' %
                         (nodesfile)],
                referenced_input_files=[ft_working_directory],
                referenced_output_files=[ft_working_directory],
                name="epac_job_key=%s" % (nodesfile),
                working_directory=ft_working_directory)
                for nodesfile in keysfile_list]
        soma_workflow = Workflow(jobs=jobs)
        if not  self.resource_id or self.resource_id == "":
            self.resource_id = socket.gethostname()
        controller = WorkflowController(self.resource_id,
                                        self.login,
                                        self.pw)
        ## run soma-workflow
        ## =================
        wf_id = controller.submit_workflow(workflow=soma_workflow,
                                           name="epac workflow")
        Helper.transfer_input_files(wf_id, controller)
        Helper.wait_workflow(wf_id, controller)
        Helper.transfer_output_files(wf_id, controller)
        if self.remove_finished_wf:
            controller.delete_workflow(wf_id)
        ## read result tree
        ## ================
        self.tree_root = store.load()
        os.chdir(cur_work_dir)
        if os.path.isdir(tmp_work_dir_path) and self.remove_local_tree:
            shutil.rmtree(tmp_work_dir_path)
        return self.tree_root

    def export_to_gui(self, soma_workflow_dirpath, **Xy):
        '''
        Example
        -------
        see the directory of "examples/run_somaworkflow_gui.py" in epac
        '''
        try:
            from soma.workflow.client import Job, Workflow
            from soma.workflow.client import Helper, FileTransfer
        except ImportError:
            errmsg = "No soma-workflow is found. "\
                "Please verify your soma-worklow"\
                "on your computer (e.g. PYTHONPATH) \n"
            sys.stderr.write(errmsg)
            sys.stdout.write(errmsg)
            raise NoSomaWFError
        if not os.path.exists(soma_workflow_dirpath):
            os.makedirs(soma_workflow_dirpath)
        tmp_work_dir_path = soma_workflow_dirpath
        cur_work_dir = os.getcwd()
        os.chdir(tmp_work_dir_path)
        ft_working_directory = FileTransfer(is_input=True,
                                        client_path=tmp_work_dir_path,
                                        name="working directory")
        ## Save the database and tree to working directory
        ## ===============================================
        np.savez(os.path.join(tmp_work_dir_path,
                 SomaWorkflowEngine.dataset_relative_path), **Xy)
        store = StoreFs(dirpath=os.path.join(
            tmp_work_dir_path,
            SomaWorkflowEngine.tree_root_relative_path))
        self.tree_root.save_tree(store=store)
        ## Subtree job allocation on disk
        ## ==============================
        node_input = NodesInput(self.tree_root.get_key())
        split_node_input = SplitNodesInput(self.tree_root,
                                           num_processes=self.num_processes)
        nodesinput_list = split_node_input.split(node_input)
        keysfile_list = self._save_job_list(tmp_work_dir_path,
                                            nodesinput_list)
        ## Build soma-workflow
        ## ===================
        jobs = [Job(command=[u"epac_mapper",
                         u'--datasets', '"%s"' %
                         (SomaWorkflowEngine.dataset_relative_path),
                         u'--keysfile', '"%s"' %
                         (nodesfile)],
                referenced_input_files=[ft_working_directory],
                referenced_output_files=[ft_working_directory],
                name="epac_job_key=%s" % (nodesfile),
                working_directory=ft_working_directory)
                for nodesfile in keysfile_list]
        soma_workflow = Workflow(jobs=jobs)
        if soma_workflow_dirpath and soma_workflow_dirpath != "":
            out_soma_workflow_file = os.path.join(soma_workflow_dirpath,
                         SomaWorkflowEngine.open_me_by_soma_workflow_gui)
            Helper.serialize(out_soma_workflow_file, soma_workflow)
        os.chdir(cur_work_dir)

    @staticmethod
    def load_from_gui(soma_workflow_dirpath):
        '''
        Result tree can be loaded from the working directory
        (soma_workflow_dirpath).
        '''
        store = StoreFs(dirpath=os.path.join(
            soma_workflow_dirpath,
            LocalEngine.tree_root_relative_path))
        tree_root = store.load()
        return tree_root

if __name__ == "__main__":
    import doctest
    doctest.testmod()
