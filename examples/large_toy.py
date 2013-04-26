# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 10:06:54 2013

@author: edouard.duchesnay@cea.fr
@author: benoit.da_mota@inria.fr
"""

from sklearn import datasets
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
X, y = datasets.make_classification(n_samples=100, n_features=500,
                                    n_informative=5)


## Realistic example
from epac import ParPerm, ParCV, WF, ParCVGridSearchRefit, Seq, ParGrid
from epac import SummaryStat, PvalPermutations

#run epac/workflow.py

# CV + Grid search of a pipeline with a nested grid search
pipeline = ParCVGridSearchRefit(*[Seq(SelectKBest(k=k),
                      ParGrid(*[SVC(kernel="linear", C=C)\
                          for C in [.1, 1, 10]]))
                for k in [1, 5, 10]],
           n_folds=5, y=y)

#pipeline.fit_predict(X=X, y=y)

wf = ParPerm(ParCV(pipeline, n_folds=3, reducer=SummaryStat(filter_out_others=False)),
                    n_perms=3, permute="y", y=y, reducer=PvalPermutations(filter_out_others=False))
wf.fit_predict(X=X, y=y)
wf.reduce()

wf.get_node(regexp="ParPerm/*/ParCV/*/ParCVGridSearchRefit/ParMethods/SelectKBest*/SVC*")



#from epac import conf, debug
conf.DEBUG = True  # set debug to True
conf.TRACE_TOPDOWN = True
wf.fit_predict(X=X, y=y)
#debug.current =None
debug.Xy
debug.current.fit_predict(**debug.Xy)

# Save tree
import tempfile
wf.save(store=tempfile.mktemp())
# Fit & Predict
# Save results
wf.save(attr="results")
key = perms_cv_lda.get_key()
# Reload tree, all you need to know is the key
tree = WF.load(key)
# Reduces results
tree.reduce()
