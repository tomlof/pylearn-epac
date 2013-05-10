#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 12:13:11 2013

@author: edouard.duchesnay@cea.fr
@author: benoit.da_mota@inria.fr
@author: jinpeng.li@cea.fr

Introduction
------------

The library Epac can create an Epac tree for machine learning algorithms.
This example shows how to export the Epac tree to soma-workflow jobs.
After the exportation, you can use soma_workflow_gui to run your jobs,
and then run the reduce process to get results.

"""

import os
import sys
import shutil
import numpy as np


from epac.export_multi_processes import export2somaworkflow
from epac.workflow.base import conf


#############################################################################
## Working directory, input and output paths
'''
+ my_working_directory
  - epac_datasets.npz
  + storekeys
    + ...
  - epac_workflow_example
'''

## Setup a working directory (my_working_directory)
## All the data should be saved and run in this directory
my_working_directory = "/tmp/my_working_directory"

## All the file paths should be ***RELATIVE*** path in the working directory
## Because the file path in the client side and that in the sever side
## are DIFFERENT.

# Training and test data
datasets_file = "./epac_datasets.npz"
# root key for Epac tree
key_file = "./storekeys"
# Soma-workflow file which can be opened by soma_workflow_gui
soma_workflow_file = "./epac_workflow_example"


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

pipeline = Seq(SelectKBest(k=2),
               ParGrid(*[SVC(kernel="linear", C=C) for C in [1, 10]]))

wf = ParPerm(ParCV(pipeline, n_folds=3),
             n_perms=10, permute="y", y=y)

# The Epac tree will be saved in with "key_file"
wf.save(store=key_file)

wf_key = wf.get_key()


#########################################################################
## RUN with soma-workflow on the local computer
## (c.f. http://brainvisa.info/soma/soma-workflow/)

# If you know nothing about your machine, you can run this code.
# After that you can run $ soma_workflow_gui
# 1. Open the file "soma_workflow_file",
# 2. Submit
# 3. Transfer input files
# 4. Waiting to finish
# 5. Transfer output files

export2somaworkflow(
    in_datasets_file=datasets_file,
    in_working_directory=my_working_directory,
    in_tree_root=wf,
    out_soma_workflow_file=soma_workflow_file
)

## If you know how many cores on your machine
## you can specify in_num_cores to optimize workload of soma-workflow
## For instance, run soma-workflow on the 4-core machine
# export2somaworkflow(
#    in_datasets_file=datasets_file,
#    in_working_directory=my_working_directory,
#    in_tree_root=wf,
#    in_num_cores=4
#    out_soma_workflow_file=soma_workflow_file,
#)

##########################################################################
### Reduce Process
### Get results using reduce after soma_workflow finish all the jobs.
#
# os.chdir(my_working_directory)
#
###### wf_key depends on your output in your map process
# wf_key = (
#    conf.KEY_PROT_FS +
#    conf.KEY_PROT_PATH_SEP +
#    key_file +
#    os.path.sep+os.walk(key_file).next()[1][0]  # to get the root node
#)
## Load Epac tree
# wf = WF.load(wf_key)
#
# wf.reduce()
