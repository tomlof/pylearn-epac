# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 11:13:20 2013

@author: jinpeng.li@cea.fr
@author: edouard.duchesnay@cea.fr
@author: benoit.da_mota@inria.fr

"""

import sys
import inspect
from abc import ABCMeta, abstractmethod
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest

from epac import Pipe, CV, Perms, Methods, CVBestSearchRefit, range_log2

current_module = sys.modules[__name__]


class WorkflowExample(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_workflow(self):
        """Retrieve workflow"""
        return None


class WFExample1(WorkflowExample):

    def get_workflow(self):
        wf = Methods(*[SVC(kernel="linear", C=C) for C in [1, 3]])
        return wf


class WFExample2(WorkflowExample):

    def get_workflow(self):
        ####################################################################
        ## EPAC WORKFLOW
        # -------------------------------------
        #             Perms                      Perm (Splitter)
        #         /     |       \
        #        0      1       2                Samples (Slicer)
        #        |
        #       CV                               CV (Splitter)
        #  /       |       \
        # 0        1       2                     Folds (Slicer)
        # |        |       |
        # Pipeline     Pipeline     Pipeline     Sequence
        # |
        # 2                                      SelectKBest (Estimator)
        # |
        # Methods
        # |                     \
        # SVM(linear,C=1)   SVM(linear,C=10)     Classifiers (Estimator)
        pipeline = Pipe(SelectKBest(k=2),
                        Methods(*[SVC(kernel="linear", C=C)
                        for C in [1, 3]]))
        wf = Perms(CV(pipeline, n_folds=3),
                        n_perms=3,
                        permute="y")
        return wf


class WFExample3(WorkflowExample):

    def get_workflow(self, n_features=int(1E03)):
        random_state = 0
        C_values = [1, 10]
        k_values = 0
        k_max = "auto"
        n_folds_nested = 5
        n_folds = 10
        n_perms = 10
        if k_max != "auto":
            k_values = range_log2(np.minimum(int(k_max), n_features),
                                  add_n=True)
        else:
            k_values = range_log2(n_features, add_n=True)
        cls = Methods(*[Pipe(SelectKBest(k=k), SVC(C=C, kernel="linear"))
                                   for C in C_values
                                   for k in k_values])
        pipeline = CVBestSearchRefit(cls,
                                     n_folds=n_folds_nested,
                                     random_state=random_state)
        wf = Perms(CV(pipeline, n_folds=n_folds),
                        n_perms=n_perms,
                        permute="y",
                        random_state=random_state)
        return wf


def get_wf_example_classes(base_class="WorkflowExample"):
    list_sub_classes = []
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj) \
        and issubclass(obj, eval(base_class)) \
        and name != base_class:
            list_sub_classes.append(obj)
#            print "==============================="
#            print issubclass(obj, eval(base_class))
#            print obj.__class__.__name__
#            print name
#            print obj
    return list_sub_classes

if __name__ == '__main__':
    list_sub_classes = get_wf_example_classes(base_class="WorkflowExample")
    for sub_class in list_sub_classes:
        print repr(sub_class)
        print sub_class().get_workflow()