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
import tempfile
import numpy as np

##############################################################################
## Input paths

## Setup a working directory (my_working_directory)
my_working_directory="/tmp/my_epac_working_directory"

## key_file and datasets_file should be RELATIVE path
## it is mantory for mapping path in soma-workflow
## since my_working_directory will be changed on the cluster
datasets_file = "./epac_datasets.npz"
key_file="./storekeys"
soma_workflow_file="./epac_workflow_example"


os.chdir(my_working_directory)

if not os.path.isdir(my_working_directory):
    os.mkdir(my_working_directory)


##############################################################################
## DATASET
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
X, y = datasets.make_classification(n_samples=100, n_features=500,
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
pipeline = Seq(SelectKBest(k=2), 
               ParGrid(*[SVC(kernel="linear", C=C) for C in [1, 10]]))
wf = ParPerm(ParCV(pipeline, n_folds=3),
                    n_perms=3, permute="y", y=y)
wf.save(store=key_file)


##############################################################################
## Nodes to run
nodes = wf.get_node(regexp="*/ParPerm/*")
## You can try another level
# nodes = wf.get_node(regexp="*/ParGrid/*")


##############################################################################
## RUN without soma-workflow
# Associate 1 job to each permutation

#cmd_jobs = [u'epac_mapper --datasets="%s" --keys="%s"' % (
#                           datasets_file, node.get_key()
#                           ) for node in nodes]
#print repr(cmd_jobs)
#
#for cmd_job in cmd_jobs:
#   os.system(cmd_job)


##############################################################################
## RUN with soma-workflow on the local computer 
## (c.f. http://brainvisa.info/soma/soma-workflow/)

from soma.workflow.client import Job, Workflow, Helper, FileTransfer

my_working_directory = FileTransfer(is_input=True,
                                    client_path=my_working_directory,
                                    name="working directory")

jobs = [Job(command=[u"epac_mapper", 
                     u'--datasets', '"%s"' % (datasets_file),
                     u'--keys','"%s"'% (node.get_key())], 
                     referenced_input_files=[my_working_directory],
                     referenced_output_files=[my_working_directory],
                     name="epac_job_key=%s"%(node.get_key()),
                     working_directory=my_working_directory) for node in nodes]

dependencies = [ ]

soma_workflow = Workflow(jobs=jobs, dependencies=dependencies)

# You can save the workflow into soma_workflow_file using Helper. 
# This workflow can be opened by $ soma_workflow_gui
Helper.serialize(soma_workflow_file, soma_workflow)

# Or you can submit directly to the server using WorkflowController
# from soma.workflow.client import WorkflowController
# controller = WorkflowController("Resource id", login, password)
# controller.submit_workflow(workflow=workflow,
#                          name="epac workflow")

