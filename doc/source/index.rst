.. epac documentation master file, created by
   sphinx-quickstart on Thu Jul 11 16:27:52 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to epac's documentation!
================================

Embarrassingly Parallel Array Computing: EPAC is a machine learning workflow builder.


Contents
========

.. toctree::

   examples.rst
   api.rst


Main Features
=============

- Easily building machine learning workflow that can be executed in sequential or in parallel.

To quick start with eapc, here is a simple example to do an embarrassing machine learning computing: 
permutation, cross-validation, and LDA classification.
We will introduce more details and examples in ????.

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

- Run epac tree in parallel on local multi-core machine or on HPC (using DRMAA).

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


- Design your own machine learning algorithm as a plug-in in epac tree.

::

  Todo......

.. toctree::
   :maxdepth: 2

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

