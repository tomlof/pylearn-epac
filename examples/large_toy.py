# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 10:06:54 2013

@author: edouard.duchesnay@cea.fr
@author: benoit.da_mota@inria.fr
"""
import sys
import optparse
import time

from sklearn import datasets
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from epac import range_log2

def do_all(**kwargs):
    random_state = 0
    k_values = range_log2(n_features, add_n=True)
    C_values = [1, 10]
    X, y = datasets.make_classification(n_samples=n_samples, n_features=n_features,
                                        n_informative=n_informative)
    # ===================
    # = With EPAC
    # ===================
    from epac import ParPerm, ParCV, ParCVGridSearchRefit, Seq, ParGrid
    from epac import SummaryStat, PvalPermutations
    
    time_start = time.time()
    ## CV + Grid search of a pipeline with a nested grid search
    pipeline = ParCVGridSearchRefit(*[
                  Seq(SelectKBest(k=k),
                      ParGrid(*[SVC(kernel="linear", C=C) for C in C_values]))
                  for k in k_values],
                  n_folds=n_folds_nested, y=y, random_state=random_state)
    wf = ParPerm(
             ParCV(pipeline,
                   n_folds=n_folds,
                   reducer=SummaryStat(filter_out_others=False)),
             n_perms=n_perms, permute="y", y=y,
             reducer=PvalPermutations(filter_out_others=False),
             random_state=random_state)
    
    time_fit_predict = time.time()
    print "Time ellapsed, tree construction:", time_fit_predict - time_start
    
    wf.fit_predict(X=X, y=y)
    time_reduce = time.time()
    print "Time ellapsed, fit predict:",  time_reduce - time_fit_predict
    
    wf.reduce()
    time_end = time.time()
    print "Time ellapsed, reduce:",   time_end - time_reduce

if __name__ == "__main__":
    # Set default values to parameters
    n_samples = 100
    n_features = int(1E03)
    n_informative = 5
    n_perms = 1000
    n_folds = 10
    n_folds_nested = 5
    # parse command line options
    parser = optparse.OptionParser()
    parser.add_option('-n', '--n_samples', help='(default %d)' % n_samples, default=n_samples)
    parser.add_option('-p', '--n_features', help='(default %d)' % n_features, default=n_features)
    parser.add_option('-m', '--n_perms', help='(default %d)' % n_perms, default=n_perms)
    parser.add_option('-c', '--n_folds', help='(default %d)' % n_folds, default=n_folds)
    parser.add_option('-i', '--n_folds_nested', help='(default %d)' % n_folds_nested, default=n_folds_nested)
    #argv = ['examples/large_toy.py', '--n_perms=10']
    #options, args = parser.parse_args(argv)
    options, args = parser.parse_args(sys.argv)
    print sys.argv
    #print options
    print options, type(options)#, {k: options[k] for k in options}
    #do_all(**options)
##python -m cProfile examples/large_toy.py >/tmp/large_toy_1000perm-10cv-5cv-1000p-100n.csv