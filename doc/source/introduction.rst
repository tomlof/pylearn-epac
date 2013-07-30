.. _introduction:

Building epac tree
==================

Easily building machine learning workflow that can be executed in sequential or in parallel.

To quick start with eapc, here is a simple example to do an embarrassing machine learning computing:
permutation, cross-validation, and LDA classification.
We will introduce more details and examples in :doc:`examples.rst`.

::

   from sklearn import datasets
   from sklearn.lda import LDA
   X, y = datasets.make_classification(n_samples=12, n_features=10,
                                    n_informative=2)
   # Permutations + Cross-validation
   # ----------------------------------
   #           Permutations           Permutations (Splitter)
   #         /     |       \
   #        0      1        2         Samples (Slicer)
   #       |
   #     CV                           CV (Splitter)
   #  /   |   \
   # 0    1    2                      Folds (Slicer)
   # |    |    |
   # LDA LDA LDA                      Classifier (Estimator)
   from epac import Perms, CV
   perms_cv_lda = Perms(CV(LDA(), n_folds=3),
                          n_perms=3, permute="y")
   # Run: Top-down process
   #   1: Permutations (shuffling X and y)
   #   2: CV (Splitting X and y into training and test parts)
   #   3: LDA (Classifilication process)
   perms_cv_lda.run(X=X, y=y)
   # Reduce: Bottom-up process
   #   1: CV (computing recognition scores from its leaves)
   #   2: Permutations (computing p values)
   perms_cv_lda.reduce()

Run in parallel
===============

Run epac tree in parallel on local multi-core machine or on HPC (using DRMAA).

::

   # Run epac tree on a multi-core machine
   from epac import SomaWorkflowEngine
   sfw_engine = SomaWorkflowEngine(
                       tree_root=perms_cv_svm,
                       num_processes=2)
   perms_cv_svm = sfw_engine.run(X=X, y=y)
   perms_cv_svm.reduce()

   # Run epac tree using soma-workflow which can be run on HPC (using DRMAA).
   from epac import SomaWorkflowEngine
   sfw_engine = SomaWorkflowEngine(
                       tree_root=perms_cv_svm,
                       num_processes=2)
   perms_cv_svm = sfw_engine.run(X=X, y=y)
   perms_cv_svm.reduce()


Design your own plug-in
=======================

Design your own machine learning algorithm as a plug-in in epac tree.

::

   from sklearn.metrics import precision_recall_fscore_support
   from sklearn.svm import SVC
   from epac.map_reduce.reducers import Reducer 
   from epac import Methods
   
   ## 1) Design your classifier
   ## =========================
   class MySVC:
       def __init__(self, C=1.0):
           self.C = C
       def transform(self, X, y):
           svc = SVC(C=self.C)
           svc.fit(X, y)
           # "transform" should return a dictionary
           return {"y/pred": svc.predict(X), "y": y}

   ## 2) Design your reducer which recall rate
   ## ========================================
   class MyReducer(Reducer):
       def reduce(self, result):
           pred_list = []
           # iterate all the results of each classifier
           # then you can design you own reducer!
           for res in result:
               precision, recall, f1_score, support = \
                       precision_recall_fscore_support(res['y'], res['y/pred'])
               pred_list.append({res['key']: recall})
           return pred_list

   ## 3) Build a tree, and then compute results 
   ## =========================================
   my_svc1 = MySVC(C=1.0)
   my_svc2 = MySVC(C=2.0)
   two_svc = Methods(my_svc1, my_svc2)
   two_svc.reducer = MyReducer()
   #           Methods
   #          /      \
   # MySVC(C=1.0)  MySVC(C=2.0) 
   # top-down process to call transform
   two_svc.top_down(X=X, y=y)
   # buttom-up process to compute scores
   two_svc.reduce()


You can get results:
[{'MySVC(C=1.0)': array([ 1.,  1.])}, {'MySVC(C=2.0)': array([ 1.,  1.])}]


