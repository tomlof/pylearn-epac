.. epac documentation master file, created by
   sphinx-quickstart on Thu Jul 11 16:27:52 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to epac's documentation!
================================

Embarrassingly Parallel Array Computing: EPAC is a machine learning workflow builder.

Contents
========


Main Features
=============

- Easily build machine learning workflow that can be executed in sequential or in parallel.

::

  X, y = datasets.make_classification(n_samples=20, n_features=5,
                                            n_informative=2)
  n_folds = 2

  # = With EPAC
  wf = CV(SVC(kernel="linear"), n_folds=n_folds,
          reducer=ClassificationReport(keep=True))
  r_epac = wf.run(X=X, y=y)

  # = With SKLEARN
  clf = SVC(kernel="linear")
  r_sklearn = list()
  for idx_train, idx_test in StratifiedKFold(y=y, n_folds=n_folds):
    #idx_train, idx_test  = cv.__iter__().next()
    X_train = X[idx_train, :]
    X_test = X[idx_test, :]
    y_train = y[idx_train, :]
    clf.fit(X_train, y_train)
    r_sklearn.append(clf.predict(X_test))


- Design your own machine learning algorithm as a plug-in in epac tree.

.. toctree::
   :maxdepth: 2


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

