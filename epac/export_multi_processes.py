#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on 2 May 2013

@author: edouard.duchesnay@cea.fr
@author: benoit.da_mota@inria.fr
@author: jinpeng.li@cea.fr

"""

import sys
import os
import socket
import subprocess

from epac.errors import NoSomaWFError, NoEpacTreeRootError
from epac.configuration import conf
from epac.utils import which
from epac.workflow.estimators import CVGridSearchRefit

_classes_cannot_be_splicted = [CVGridSearchRefit().__class__.__name__]


def _is_cannot_be_splicted(signature):
    for _class_cannot_be_splicted in _classes_cannot_be_splicted:
        if _class_cannot_be_splicted in signature:
            return True
    return False


def _push_node_in_list(node, nodes_per_processor_list):
    '''Push node in the list which contains the minimun number of nodes
    '''
    min_len = -1
    min_key = -1
    for key in nodes_per_processor_list.keys():
        if (min_len == -1 or min_len > len(nodes_per_processor_list[key])):
            min_key = key
            min_len = len(nodes_per_processor_list[key])
    if min_key != -1:
        nodes_per_processor_list[min_key].append(node.get_key())
    return nodes_per_processor_list


def _export_nodes_recursively(
    node,
    num_processors,
    nodes_per_processor_list
):
    '''Allocate recursively the nodes for list
    '''
    num_processors = len(nodes_per_processor_list)
    children_nodes = node.children
    len_children = len(children_nodes)
    if len_children == 0:
        nodes_per_processor_list = _push_node_in_list(
            node,
            nodes_per_processor_list)
        return nodes_per_processor_list
    left = len_children % num_processors
    if len_children >= num_processors:
        for i in range(len_children-left):
            nodes_per_processor_list = _push_node_in_list(
                node=children_nodes[i],
                nodes_per_processor_list=nodes_per_processor_list)
    if left > 0:
        for i in range(len_children-left, len_children):
            if not _is_cannot_be_splicted(children_nodes[i].get_signature()):
                nodes_per_processor_list = _export_nodes_recursively(
                    children_nodes[i],
                    num_processors,
                    nodes_per_processor_list)
            else:
                nodes_per_processor_list = _push_node_in_list(
                    node=children_nodes[i],
                    nodes_per_processor_list=nodes_per_processor_list)
    return nodes_per_processor_list


def _export_nodes2num_processes(node, num_processors):
    '''export nodes
    Try to build "num_processors" queues which contains almost equally number
    of Epac nodes for computing.

    Parameters
    ----------
    node:epac.base.WFNode
        Epac tree root where you want to start to parallelly compute
        using "in_num_processes" cores.

    num_processors:integer
        The number of processors you have.
    '''
    nodes_per_processor_list = dict()
    for i in range(num_processors):
        nodes_per_processor_list[i] = list()
    nodes_per_processor_list = _export_nodes_recursively(
        node=node,
        num_processors=num_processors,
        nodes_per_processor_list=nodes_per_processor_list)
    return nodes_per_processor_list


def _gen_keysfile_list_from_nodes_list(
    working_directory,
    nodes_per_processor_list
):
    '''Generating a list of files where each file contains a set of keys.
    Generating a list of files where each file contains a set of keys.
    A key means a node which can be considered as a job.
    '''
    keysfile_list = list()
    jobi = 0
    for npp_key in nodes_per_processor_list.keys():
        keysfile = "."+os.path.sep+repr(jobi)+"."+conf.SUFFIX_JOB
        keysfile_list.append(keysfile)
        # print "in_working_directory="+in_working_directory
        # print "keysfile="+keysfile
        abs_keysfile = os.path.join(working_directory, keysfile)
        f = open(abs_keysfile, 'w')
        for key_signature in nodes_per_processor_list[npp_key]:
            f.write("%s\n" % key_signature)
        f.close()
        jobi = jobi + 1
    return keysfile_list


def export2somaworkflow(in_datasets_file_relative_path,
                        in_working_directory,
                        out_soma_workflow_file=None,
                        in_tree_root=None,
                        in_num_processes=15,
                        in_is_sumbit=False,
                        in_resource_id=None,
                        in_login="",
                        in_pw=""):
    """Export soma epac nodes into a soma workflow file

    Examples
    ----------

    See pylearn-epac/examples/run_somaworkflow_gui.py
    See pylearn-epac/examples/run_somaworkflow_no_gui.py

    Parameters
    ----------
    in_datasets_file_relative_path: string
        The X and y database file. This should use relative path.

    in_working_directory: string
        Since all the data (e.g. in_datasets_file_relative_path) should
        use relative paths, "in_working_directory" contains those data, and
        "in_working_directory" can be absolute path on client side.

    out_soma_workflow_file: string
        This file can be opened by soma_workflow_gui to execute the workflow
        If it is "None" or empty, the soma-workflow won't be saved.

    in_tree_root: epac.base.WFNode
        Epac tree root where you want to start to parallelly compute
        using "in_num_processes" cores.

    in_num_processes: integer
        This parameter is used for optimizing exported nodes (in_nodes)
        according to the number of processors that you have on your running
        machine. "in_num_processes" should bigger than the real number of cores
        that you have, otherwise you cannot use the full performance of your
        machine.

    in_is_sumbit: boolean
        Does this function submit the workflow to soma-workflow

    in_resource_id: string
        The resource name. When in_resource_id equal to None,
        it will use the local computer to calculate

    in_login: string
        The resource login using ssh

    in_pw: string
        The password using in_login and ssh

    Return
    ----------
    if set in_is_sumbit = True, and then this function will return

    wf_id:string
        workflow id

    controller:soma.workflow.client.WorkflowController
        soma-workflow controller

    Exception
    ----------
    epac.errors.NoSomaWFError
        The function can not find soma-workflow on the client machine.

    epac.errors.NoEpacTreeRootError
        The tree root node is none. Please make sure tree root is set.
    """
    try:
        from soma.workflow.client import Job, Workflow, Helper, FileTransfer
    except ImportError:
        errmsg = "No soma-workflow is found. Please verify your soma-worklow "\
            "on your computer (e.g. PYTHONPATH) \n"
        sys.stderr.write(errmsg)
        sys.stdout.write(errmsg)
        raise NoSomaWFError
    ft_working_directory = FileTransfer(is_input=True,
                                        client_path=in_working_directory,
                                        name="working directory")
    if not in_tree_root:
        raise NoEpacTreeRootError
    nodes_per_processor_list = _export_nodes2num_processes(
        in_tree_root,
        in_num_processes)
    keysfile_list = _gen_keysfile_list_from_nodes_list(
        in_working_directory,
        nodes_per_processor_list)
    jobs = [Job(command=[u"epac_mapper",
                         u'--datasets', '"%s"' %
                         (in_datasets_file_relative_path),
                         u'--keysfile', '"%s"' %
                         (nodesfile)],
                referenced_input_files=[ft_working_directory],
                referenced_output_files=[ft_working_directory],
                name="epac_job_key=%s" % (nodesfile),
                working_directory=ft_working_directory)
            for nodesfile in keysfile_list]
    soma_workflow = Workflow(jobs=jobs)
    if out_soma_workflow_file and out_soma_workflow_file != "":
        # You can save the workflow into out_soma_workflow_file using Helper.
        # This workflow can be opened by $ soma_workflow_gui
        Helper.serialize(out_soma_workflow_file, soma_workflow)
    if in_is_sumbit:
        from soma.workflow.client import WorkflowController
        if not in_resource_id or in_resource_id == "":
            in_resource_id = socket.gethostname()
        controller = WorkflowController(in_resource_id, in_login, in_pw)
        wf_id = controller.submit_workflow(workflow=soma_workflow,
                                           name="epac workflow")
        Helper.transfer_input_files(wf_id, controller)
        return (wf_id, controller)


def run_multi_processes(
    in_datasets_file_relative_path,
    in_working_directory,
    in_tree_root,
    in_num_processes=2,
    in_is_wait=False
):
    ''' Run Epac directly on local machine with n cores


    Examples
    ----------

    See pylearn-epac/examples/run_multi_processes.py

    Parameters
    ----------
    in_datasets_file_relative_path: string
        The X and y database file. This should use relative path.

    in_working_directory: string
        Since all the data (e.g. in_datasets_file_relative_path) should
        use relative paths, "in_working_directory" contains those data, and
        "in_working_directory" can be absolute path on client side.

    in_tree_root: epac.base.WFNode
        Epac tree root where you want to start to parallelly compute
        using "in_num_processes" cores.

    in_num_processes: integer
        This parameter is used for optimizing exported nodes (in_nodes)
        according to the number of processors that you have on your running
        machine. "in_num_processes" should bigger than the real number of cores
        that you have, otherwise you cannot use the full performance of your
        machine.

    in_is_wait: boolean
         Do we need to wait for the termination of all the processes.

    Return
    ----------
    if set in_is_wait = False, and then this function will return

    processes: list of processes
        List of processes which is "p = subprocess.Popen(args)"
    '''
    bin_epac_mapper = which("epac_mapper")
    if not bin_epac_mapper:
        raise ValueError("epac_mapper cannot be found in PATH variable. "
                         "Please verify if epac_mapper is"
                         "contained in PATH.")
    nodes_per_processor_list = _export_nodes2num_processes(
        node=in_tree_root,
        num_processors=in_num_processes
    )
    keysfile_list = _gen_keysfile_list_from_nodes_list(
        in_working_directory,
        nodes_per_processor_list
    )
    cmd_jobs = [u'epac_mapper --datasets="%s" --keysfile="%s"'
                % (in_datasets_file_relative_path, jobfile)
                for jobfile in keysfile_list]
    processes = list()
    for index in range(len(cmd_jobs)):
        cmd_job = cmd_jobs[index]
        jobfile = keysfile_list[index]
        stdout_file = os.path.join(in_working_directory, jobfile + ".stdout")
        stderr_file = os.path.join(in_working_directory, jobfile + ".stderr")
        process = None
        try:
#            sys.stdout.write("cmd_job="+cmd_job+"\n")
#            sys.stdout.write("stdout_file="+stdout_file+"\n")
#            sys.stdout.write("stderr_file="+stderr_file+"\n")
# sys.stdout.write("in_working_directory="+in_working_directory+"\n")
            stdout_file = open(stdout_file, "wb")
            stderr_file = open(stderr_file, "wb")
            cmd_job = cmd_job.split()
            process = subprocess.Popen(cmd_job,
                                       stdout=stdout_file,
                                       stderr=stderr_file,
                                       cwd=in_working_directory)
        except Exception, e:
            s = '%s: %s \n' % (type(e), e)
            sys.stderr.write(s)
            processes.append(None)
        finally:
            processes.append(process)
    if not in_is_wait:
        return processes
    else:
        for process in processes:
            process.wait()