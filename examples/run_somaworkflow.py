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
import shutil


## Reursive load
import numpy as np


from epac.exports import export2somaworkflow, exportNodes4somaworkflow
from epac.workflow.base import conf

#############################################################################
## Map process and reduce process
## Map process means we compute results for leaves
## Reduce process means we collect results from leaves.
## We should firstly run one of map processes, and secondly run the reduce 
## process
## Here you are four examples, three for map process and one for reduce 
## process

class ExampleOptions:
    # Map process: compute all the results
    option_run_without_soma_workflow="Run without soma-workflow"
    option_run_with_soma_workflow_gui="Run with soma_workflow_gui"
    option_run_with_soma_workflow_submit="Run with soma-workflow and submit"
    # Reduce process: get results
    option_reduce_tree="Reduce the tree"
    
## You can choose an example here:
ChosenExampleOption=ExampleOptions.option_run_without_soma_workflow
#ChosenExampleOption=ExampleOptions.option_run_with_soma_workflow_gui
#ChosenExampleOption=ExampleOptions.option_run_with_soma_workflow_submit
#ChosenExampleOption=ExampleOptions.option_reduce_tree


IsMapProcess=True

if ChosenExampleOption==ExampleOptions.option_reduce_tree:
    IsMapProcess=False

#############################################################################
## Working directory, input and output paths
'''
+my_epac_working_directory
  -epac_datasets.npz
  +storekeys
    +...
  -epac_workflow_example
'''

## Setup a working directory (my_working_directory)
## All the data should be saved and run in this directory
my_working_directory="/tmp/my_epac_working_directory"

## key_file and datasets_file should be ***RELATIVE*** path
## It is mandatory for mapping path operation in soma-workflow
## since my_working_directory will be changed on the cluster
datasets_file = "./epac_datasets.npz" # Training and test data
key_file="./storekeys" # Epac tree
soma_workflow_file="./epac_workflow_example" # Soma-workflow file


#############################################################################
## Clean and change the working directory 
## so that we can use relative path in the directory "my_working_directory"

if os.path.isdir(my_working_directory):
    shutil.rmtree(my_working_directory)

os.mkdir(my_working_directory)
os.chdir(my_working_directory)


#############################################################################
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
                        n_perms=10, permute="y", y=y)
       
    wf.save(store=key_file)
    
    
    wf_key=wf.get_key()
    

    print ("After the map process, you can use this key to load the "
          +"tree, and then run reduce")
          
    print "    wf_key="+wf_key
    
else:
    #########################################################################
    ## Reduce Process
    ## Get results using reduce after soma_workflow finish all the jobs.
    
    
    os.chdir(my_working_directory)
    
    ##### wf_key depends on your output in your map process
    wf_key=(
    conf.KEY_PROT_FS+
    conf.KEY_PROT_PATH_SEP+
    key_file+
    os.path.sep+os.walk(key_file).next()[1][0] # to get the root node
    )
    
    
    wf = WF.load(wf_key)
    
    if ChosenExampleOption==ExampleOptions.option_reduce_tree:
        ## Reduces results
        wf.reduce()
    
    sys.exit(0)


if ChosenExampleOption==ExampleOptions.option_run_without_soma_workflow:
    #########################################################################
    ## RUN without soma-workflow
    
    num_cores=3
    nodes_per_processor_list=exportNodes4somaworkflow(wf,num_cores)

    jobi=0
    jobfile_list=list()
    for npp_key in nodes_per_processor_list.keys():
        keysfile="."+os.path.sep+repr(jobi)+"."+conf.SUFFIX_JOB
        jobfile_list.append(keysfile)
        f = open(keysfile,'w')
        for keynode in nodes_per_processor_list[npp_key]:        
            f.write("%s\n"%keynode)
        f.close()
        jobi=jobi+1    

    cmd_jobs = [u'epac_mapper --datasets="%s" --keysfile="%s"' \
    % (datasets_file, jobfile) for jobfile in jobfile_list]
    
  
    print repr(cmd_jobs)
    
    for cmd_job in cmd_jobs:
       os.system(cmd_job)

if (ChosenExampleOption==ExampleOptions.option_run_with_soma_workflow_gui):
    #########################################################################
    ## RUN with soma-workflow on the local computer 
    ## (c.f. http://brainvisa.info/soma/soma-workflow/)

    # If you know nothing about your machine, you can run this code
    export2somaworkflow(in_datasets_file        =datasets_file,
                        in_working_directory    =my_working_directory,
                        out_soma_workflow_file  =soma_workflow_file,
                        in_tree_root            =wf
                        )

#    # If you want run your algorithm with 4 processors (in_num_cores), 
#    # you can run this code
#    export2somaworkflow(in_datasets_file        =datasets_file,
#                        in_working_directory    =my_working_directory,
#                        out_soma_workflow_file  =soma_workflow_file,
#                        in_tree_root            =wf,
#                        in_num_cores            =4
#                        )
    
#    # If you want run your algorithm with defined nodes
#    # for instance, nodes = wf.get_node(regexp="*/ParPerm/*")
#    nodes = wf.get_node(regexp="*/ParPerm/*")
#    export2somaworkflow(in_datasets_file        =datasets_file,
#                        in_working_directory    =my_working_directory,
#                        out_soma_workflow_file  =soma_workflow_file,
#                        in_nodes                =nodes
#                        )

if ChosenExampleOption==ExampleOptions.option_run_with_soma_workflow_submit:
    #########################################################################
    ## RUN with soma-workflow and submit on the local computer 
    ## in_resource_id == "" means that we use the local computer
    ## (c.f. http://brainvisa.info/soma/soma-workflow/)
    (wf_id,controller)=export2somaworkflow(
                        in_datasets_file        =datasets_file,
                        in_working_directory    =my_working_directory,
                        out_soma_workflow_file  =soma_workflow_file,
                        in_tree_root            =wf,
                        in_is_sumbit            =True,
                        in_resource_id          ="",
                        in_login                ="",
                        in_pw                   =""
                        )
                        
    from soma.workflow.client import Helper
    
    ## wait the workflow to finish
    Helper.wait_workflow(wf_id,controller)
    ## transfer the output files from the workflow
    Helper.transfer_output_files(wf_id,controller)
    ## Remove the workflow in soma-workflow
    controller.delete_workflow(wf_id)
