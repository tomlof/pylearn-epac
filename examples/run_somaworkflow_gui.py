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
import optparse
import shutil
import time
import numpy as np

from epac.export_multi_processes import export2somaworkflow

from sklearn import datasets
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest

def do_all(options):
    '''
    + my_working_directory
      - epac_datasets.npz
      + epac_tree
    '''
    ## All the file paths should be ***RELATIVE*** path in the working directory
    # Training and test data
    datasets_file = "./epac_datasets.npz"
    # root key for Epac tree
    tree_root_relative_path = "./epac_tree"
    somaworkflow_relative_path = "./soma-workflow_epac_tree"

    ## 1) Create Working directory
    ## ===========================
    if os.path.isdir(options.working_dir_path):
        shutil.rmtree(options.working_dir_path)
    os.mkdir(options.working_dir_path)
    os.chdir(options.working_dir_path)

    ## 2) Build dataset
    ## ================
    X, y = datasets.make_classification(n_samples=options.n_samples,
                                        n_features=options.n_features,
                                        n_informative=options.n_informative)
    np.savez(datasets_file, X=X, y=y)

    ## 3) Build Workflow
    ## =================
    from epac import Permutations, CV, Seq
    from epac import SummaryStat, PvalPermutations, Grid
    
    time_start = time.time()
    ##############################################################################
    ## EPAC WORKFLOW
    # -------------------------------------
    #             Permutations                Perm (Splitter)
    #         /     |       \
    #        0      1       2            Samples (Slicer)
    #        |
    #       CV                        CV (Splitter)
    #  /       |       \
    # 0        1       2                 Folds (Slicer)
    # |        |       |
    # Seq     Seq     Seq                Sequence
    # |
    # 2                                  SelectKBest (Estimator)
    # |
    # Grid
    # |                     \
    # SVM(linear,C=1)   SVM(linear,C=10)  Classifiers (Estimator)
    pipeline = Seq(SelectKBest(k=2),
                   Grid(*[SVC(kernel="linear", C=C) for C in [1, 10]]))
    wf = Permutations(
             CV(pipeline, n_folds=options.n_folds,
                   reducer=SummaryStat(filter_out_others=True)),
             n_perms=options.n_perms, permute="y", y=y,
             reducer=PvalPermutations(filter_out_others=True))

    print "Time ellapsed, tree construction:", time.time() - time_start
    time_save = time.time()
    ## 4) Save on disk
    ## ===============
    wf.save(store=tree_root_relative_path)
    print "Time ellapsed, saving on disk:",  time.time() - time_save
    ## 5) Run
    ## ======
    #wf.fit_predict(X=X, y=y)
    export2somaworkflow(
        in_datasets_file=datasets_file,
        in_working_directory=options.working_dir_path,
        in_tree_root=wf,
        out_soma_workflow_file=somaworkflow_relative_path
    )

#    print "Time ellapsed, fit predict:",  time.time() - time_fit_predict
#    wf_key = wf.get_key()
#
#    time_reduce = time.time()
#    ## 6) Load Epac tree & Reduce
#    ## ==========================
    reduce_filename = os.path.join(os.getcwd(), "reduce.py")
    f = open(reduce_filename, 'w')
    reduce_str = """from epac import WF
import os
os.chdir("%s")
wf = WF.load("%s")
print wf.reduce()""" % (options.working_dir_path, wf.get_key())
    f.write(reduce_str)
    f.close()
    print "#First run\nsoma_workflow_gui\n#When done run:\npython %s" % reduce_filename
#    print "Time ellapsed, reduce:",   time.time() - time_reduce


if __name__ == "__main__":
    # Set default values to parameters
    n_samples = 100
    n_features = int(1E03)
    n_informative = 5
    n_perms = 10
    n_folds = 10
    n_folds_nested = 5
    k_max = "auto"
    working_dir_path = "/tmp/my_working_directory"

    # parse command line options
    parser = optparse.OptionParser()
    parser.add_option('-n', '--n_samples',
        help='(default %d)' % n_samples, default=n_samples, type="int")
    parser.add_option('-p', '--n_features',
        help='(default %d)' % n_features, default=n_features, type="int")
    parser.add_option('-i', '--n_informative',
        help='(default %d)' % n_informative, default=n_informative, type="int")
    parser.add_option('-m', '--n_perms',
        help='(default %d)' % n_perms, default=n_perms, type="int")
    parser.add_option('-f', '--n_folds',
        help='(default %d)' % n_folds, default=n_folds, type="int")
    parser.add_option('-g', '--n_folds_nested',
        help='(default %d)' % n_folds_nested, default=n_folds_nested, type="int")
    parser.add_option('-k', '--k_max',
        help='"auto": 1, 2, 4, ... n_features values. "fixed": 1, 2, 4, ..., k_max (default %s)' % k_max, default=k_max, type="string")
    parser.add_option('-t', '--trace',
        help='Trace execution (default %s)' % False, action='store_true', default=False)
    parser.add_option('-w', '--working_dir_path',
        help='(default %s)' % working_dir_path, default=working_dir_path)
    #argv = []
    #options, args = parser.parse_args(argv)
    options, args = parser.parse_args(sys.argv)
    do_all(options)