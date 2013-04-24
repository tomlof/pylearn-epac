# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 12:13:11 2013

@author: edouard.duchesnay@cea.fr
@author: benoit.da_mota@inria.fr
"""

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
datasets_file = tempfile.NamedTemporaryFile(suffix="_datasets.npz")
np.savez(datasets_file, X=X, y=y)
datasets_file.name

##############################################################################
## EPAC WORKFLOW
from epac import ParPerm, ParCV, WF, Seq, ParGrid
pipeline = Seq(SelectKBest(k=2), ParGrid(*[SVC(kernel="linear", C=C) for C in [1, 10]]))
wf = ParPerm(ParCV(pipeline, n_folds=3),
                    n_perms=3, permute="y", y=y)
wf.save(store=tempfile.mktemp())

##############################################################################
## RUN
# Associate 1 job to each permutation
nodes = wf.get_node(regexp="*/ParPerm/*")

jobs = [u'./epac_mapper --datasets="%s" --keys="%s"' % (datasets_file.name, node.get_key()) for node in nodes]
print jobs[0]
