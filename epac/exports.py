#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on 2 May 2013

@author: edouard.duchesnay@cea.fr
@author: benoit.da_mota@inria.fr
@author: jinpeng.li@cea.fr

"""

import sys
from epac.errors import NoSomaWFError
import socket

def export2somaworkflow(in_datasets_file, 
                        in_working_directory, 
                        in_nodes, 
                        out_soma_workflow_file,
                        in_is_sumbit=False,
                        in_resource_id=None,
                        in_login="",
                        in_pw=""
                        ):
    """Export soma epac nodes into a soma workflow file

    Parameters
    ----------
    in_datasets_file: string
        The X and y database file
    
    in_working_directory: string
        Since all the data (e.g. in_datasets_file) should use relative paths, 
        "in_working_directory" contains those data 
        , and "in_working_directory" can be absolute path on client side
    
    in_nodes: a list of epac.base.WFNode
        Since all the data (e.g. in_datasets_file) should use relative paths, 
        "in_working_directory" contains those data 
        , and "in_working_directory" can be absolute path on client side
    
    out_soma_workflow_file: string
        This file can be opened by soma_workflow_gui to execute the workflow
    
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
        
    """ 
    try:
      from soma.workflow.client import Job, Workflow, Helper, FileTransfer
    except ImportError:
      errmsg="No soma-workflow is found. Please verify your soma-worklow "\
      "on your computer (e.g. PYTHONPATH) \n"
      sys.stderr.write(errmsg)
      sys.stdout.write(errmsg)
      raise NoSomaWFError
      
    
    in_working_directory = FileTransfer(is_input=True,
                                        client_path=in_working_directory,
                                        name="working directory")
    
    jobs = [Job(command=[u"epac_mapper", 
                         u'--datasets', '"%s"' % (in_datasets_file),
                         u'--keys','"%s"'% (node.get_key())], 
                         referenced_input_files=[in_working_directory],
                         referenced_output_files=[in_working_directory],
                         name="epac_job_key=%s"%(node.get_key()),
                         working_directory=in_working_directory) for node in in_nodes]
    
    soma_workflow = Workflow(jobs=jobs)
    
    # You can save the workflow into out_soma_workflow_file using Helper. 
    # This workflow can be opened by $ soma_workflow_gui
    Helper.serialize(out_soma_workflow_file, soma_workflow)
    
    if in_is_sumbit:
        
        from soma.workflow.client import WorkflowController
        
        if not in_resource_id or in_resource_id == "":
            in_resource_id=socket.gethostname()
        
        controller = WorkflowController(in_resource_id, in_login, in_pw)
        
        wf_id=controller.submit_workflow(workflow=soma_workflow,
                                  name="epac workflow")
                                  
        Helper.transfer_input_files(wf_id, controller)
        
        return (wf_id,controller)