#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 12:13:11 2013

@author: edouard.duchesnay@cea.fr
@author: benoit.da_mota@inria.fr
@author: jinpeng.li@cea.fr

"""

import os
import sys

## Reursive load
import numpy as np


from epac.exports import export2somaworkflow

##############################################################################
## Map process and reduce process

class ExampleOptions:
    # Map process: compute all the results
    option_map_process="Using map process"
    option_run_without_soma_workflow="Run without soma-workflow"
    option_run_with_soma_workflow_gui="Run with soma_workflow_gui"
    option_run_with_soma_workflow_code="Run with soma-workflow using code"
    # Reduce process: get results
    option_reduce_tree="Reduce the tree"
    
## You can choose an example here:
    
#ChosenExampleOption=ExampleOptions.option_run_without_soma_workflow
ChosenExampleOption=ExampleOptions.option_run_with_soma_workflow_gui
#ChosenExampleOption=ExampleOptions.option_run_with_soma_workflow_code
#ChosenExampleOption=ExampleOptions.option_reduce_tree


IsMapProcess=True

if ChosenExampleOption==ExampleOptions.option_reduce_tree:
    IsMapProcess=False


##############################################################################
## Working directory and input paths
'''
+my_epac_working_directory
  -epac_datasets.npz
  +storekeys
    +...
  -epac_workflow_example
'''

## Setup a working directory (my_working_directory)
my_working_directory="/tmp/my_epac_working_directory"

## key_file and datasets_file should be ***RELATIVE*** path
## It is mandatory for mapping path in soma-workflow
## since my_working_directory will be changed on the cluster
datasets_file = "./epac_datasets.npz"
key_file="./storekeys"
soma_workflow_file="./epac_workflow_example"


##############################################################################
## Change the working directory 
## so that we can use relative path in the directory my_working_directory

if not os.path.isdir(my_working_directory):
    os.mkdir(my_working_directory)

os.chdir(my_working_directory)


##############################################################################
## DATASET
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest


X=None
y=None

if not IsMapProcess:
    db=np.load(datasets_file)
    X=db["X"]
    y=db["y"]
else:   
    X, y = datasets.make_classification(n_samples=10000, n_features=500,
                                        n_informative=5)                   
    np.savez(datasets_file, X=X, y=y)


##############################################################################
## EPAC WORKFLOW
# -------------------------------------
#             ParPerm                Perm (Splitter)
#         /     |       \
#        0      1       2            Samples (Slicer)
#        |
#       ParCV                        CV (Splitter)
#  /       |       \
# 0        1       2                 Folds (Slicer)
# |        |       |
# Seq     Seq     Seq                Sequence
# |
# 2                                  SelectKBest (Estimator)
# |
# ParGrid
# |                     \
# SVM(linear,C=1)   SVM(linear,C=10)  Classifiers (Estimator)


from epac import ParPerm, ParCV, WF, Seq, ParGrid


wf=None

if IsMapProcess:
    
    pipeline = Seq(SelectKBest(k=2), 
                   ParGrid(*[SVC(kernel="linear", C=C) for C in [1, 10]]))
                   
    wf = ParPerm(ParCV(pipeline, n_folds=3),
                        n_perms=100, permute="y", y=y)
                        
    wf.save(store=key_file)
    
    wf_key=wf.get_key()

    print ("### After the map process, you can use this key to load the "
          +"tree, and then run reduce")
          
    print "wf_key="+wf_key
    
else:
    #########################################################################
    ## Reduce Process
    ## Get results using reduce after soma_workflow finish all the jobs.
    
    from epac.workflow.base import conf
    
    os.chdir(my_working_directory)
    
    ##### wf_key depends on your output in your map process
    wf_key=(
    conf.KEY_PROT_FS+
    conf.KEY_PROT_PATH_SEP+
    key_file+
    os.path.sep+os.walk(key_file).next()[1][0]
    )
    
    wf = WF.load(wf_key)
    
    if ChosenExampleOption==ExampleOptions.option_reduce_tree:
        ## Reduces results
        wf.reduce()


##############################################################################
## Nodes to run
# nodes = wf.get_node(regexp="*/ParPerm/*")
## You can try another level
nodes = wf.get_node(regexp="*/ParGrid/*")



if ChosenExampleOption==ExampleOptions.option_run_without_soma_workflow:
    ##############################################################################
    ## RUN without soma-workflow
    # Associate 1 job to each permutation
    
    cmd_jobs = [u'epac_mapper --datasets="%s" --keys="%s"' % (
                               datasets_file, node.get_key()
                               ) for node in nodes]
    
    print repr(cmd_jobs)
    
    for cmd_job in cmd_jobs:
       os.system(cmd_job)

if (ChosenExampleOption==ExampleOptions.option_run_with_soma_workflow_gui 
   or ChosenExampleOption==ExampleOptions.option_run_with_soma_workflow_code):
    ##############################################################################
    ## RUN with soma-workflow on the local computer 
    ## (c.f. http://brainvisa.info/soma/soma-workflow/)
    
    
    export2somaworkflow(datasets_file, my_working_directory, nodes, soma_workflow_file)
    
    
 
#    if ChosenExampleOption==ExampleOptions.option_run_with_soma_workflow_code:
#        
#        # Or you can submit directly to the server using WorkflowController
#        from soma.workflow.client import WorkflowController
#        login="xxxxx"        
#        password="xxxxx"
#        controller = WorkflowController("Resource id", login, password)
#        wf_id=controller.submit_workflow(workflow=soma_workflow,
#                                  name="epac workflow")
#        Helper.transfer_input_files(wf_id, controller)



