epac
----

Embarrassingly Parallel Array Computing: EPAC is a machine learning workflow
builder.

* You can build a big machine workflow:

```

    Permutation (Perm) + Cross-validation (CV) of SVM(linear) and SVM(rbf)
    ----------------------------------------------------------------------
             Perms         Perm (Splitter)
         /     |       \
        0      1       2   Samples
               |
              CV           CV (Splitter)
          /   |   \
         0    1    2       Folds
              |
           Methods         Methods (Splitter)
       /           \
    SVM(linear)  SVM(rbf)  Classifiers (Estimator)

```

using very simple codes:


```python

    from sklearn.svm import SVC
    from epac import Perms, CV, Methods
    perms_cv_svm = Perms(CV(
                     Methods(*[SVC(kernel="linear"), SVC(kernel="rbf")]),
                     n_folds=3),
                     n_perms=3)
    perms_cv_svm.run(X=X, y=y)
    perms_cv_svm.reduce() # get results

```


* Run epac tree in parallel on local multi-core machine or even on HPC using soma-workflow.

```python

    from epac import LocalEngine
    local_engine = LocalEngine(tree_root=perms_cv_svm, num_processes=2)
    perms_cv_svm = local_engine.run(X=X, y=y)
    perms_cv_svm.reduce()

```

* Design your own machine learning algorithm as a plug-in node in epac tree.

```python

    todo

```

Others
------

**Installation**
http://neurospin.github.io/pylearn-epac/installation.html

**Tutorials**
http://neurospin.github.io/pylearn-epac/tutorials.html

**Documentation**
http://neurospin.github.io/pylearn-epac


