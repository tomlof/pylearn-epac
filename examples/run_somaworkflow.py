# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 12:13:11 2013

@author: edouard.duchesnay@cea.fr
@author: benoit.da_mota@inria.fr
@author: jinpeng.li@cea.fr
"""

import os


## Reursive load
import tempfile
import numpy as np

##############################################################################
## DATASET
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
X, y = datasets.make_classification(n_samples=100, n_features=500,
                                    n_informative=5)
datasets_file = tempfile.NamedTemporaryFile(suffix="_datasets.npz",delete=False)
np.savez(datasets_file, X=X, y=y)
datasets_file.name

##############################################################################
## EPAC WORKFLOW
# -------------------------------------
#             ParPerm                Perm (Splitter)
#         /     |       \
#        0      1       2            Samples (Slicer)
#        |
#      ParCV                         CV (Splitter)
#  /       |       \
# 0        1       2                 Folds (Slicer)
# |        |       |
# Seq     Seq     Seq                Sequence
# |
# 2                                  SelectKBest (Estimator)
# |
# ParGrid
# |                \
# SVM(linear,C=1)   SVM(linear,C=10)  Classifiers (Estimator)

from epac import ParPerm, ParCV, WF, Seq, ParGrid
pipeline = Seq(SelectKBest(k=2), 
               ParGrid(*[SVC(kernel="linear", C=C) for C in [1, 10]]))
wf = ParPerm(ParCV(pipeline, n_folds=3),
                    n_perms=3, permute="y", y=y)
wf.save(store=tempfile.mktemp())


##############################################################################
## Nodes to run
nodes = wf.get_node(regexp="*/ParPerm/*")

## You can try another level
# nodes = wf.get_node(regexp="*/ParGrid/*")


##############################################################################
## RUN without soma-workflow
# Associate 1 job to each permutation

# cmd_jobs = [u'epac_mapper --datasets="%s" --keys="%s"' % (
#                            datasets_file.name, node.get_key()
#                            ) for node in nodes]

# for cmd_job in cmd_jobs:
#    os.system(cmd_job)


##############################################################################
## RUN with soma-workflow on the local computer (c.f. http://brainvisa.info/soma/soma-workflow/)

from soma.workflow.client import Job, Workflow, Helper

jobs = [Job(command=[u"epac_mapper", 
                     u'--datasets', '"%s"' % (datasets_file.name),
                     u'--keys','"%s"'% (node.get_key())], 
                    name="epac_job_key=%s"%(node.get_key())) for node in nodes]

dependencies = [ ]

soma_workflow = Workflow(jobs=jobs, dependencies=dependencies)

# You can save the workflow into "/tmp/epac_workflow_example" using Helper. 
# This workflow can be opened by $ soma_workflow_gui
Helper.serialize("/tmp/epac_workflow_example", soma_workflow)


# Or you can submit directly to the server using WorkflowController
# from soma.workflow.client import WorkflowController
# controller = WorkflowController("Resource id", login, password)
# controller.submit_workflow(workflow=workflow,
#                          name="epac workflow")

