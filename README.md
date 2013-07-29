epac
----

Embarrassingly Parallel Array Computing: EPAC is a machine learning workflow
builder.


Given a database:
 
```python

    from sklearn import datasets
    X, y = datasets.make_classification(n_samples=12,
                                        n_features=10,
                                        n_informative=2,
                                        random_state=1)

```



* You can build a big machine workflow:

```

    Permutation (Perm) + Cross-validation (CV) of SVM(linear) and SVM(rbf)
    ----------------------------------------------------------------------
             Perms          Perm (Splitter)
        /     |       \
       0      1       2     Samples
              |
              CV            CV (Splitter)
          /   |   \
         0    1    2        Folds
              |
           Methods          Methods (Splitter)
       /           \
    SVM(linear)  SVM(rbf)   Classifiers (Estimator)

```

using very simple codes:


```python

    from sklearn.svm import SVC
    from epac import Perms, CV, Methods
    perms_cv_svm = Perms(CV(
                     Methods(*[SVC(kernel="linear"), SVC(kernel="rbf")]),
                     n_folds=3),
                     n_perms=3)
    perms_cv_svm.run(X=X, y=y) # Top-down process: computing recognition rates, etc.
    perms_cv_svm.reduce() # Bottom-up process: computing p-values, etc.

```

Then you can get results like:

```
ResultSet(
[{'key': SVC(kernel=linear), 'y/test/score_f1': [ 0.5  0.5], 'y/test/score_recall_mean/pval': [ 0.5], 'y/test/score_recall/pval': [ 0.5  0.5], 'y/test/score_accuracy/pval': [ 0.5], 'y/test/score_f1/pval': [ 0.5  0.5], 'y/test/score_precision/pval': [ 0.5  0.5], 'y/test/score_precision': [ 0.5  0.5], 'y/test/score_recall': [ 0.5  0.5], 'y/test/score_accuracy': 0.5, 'y/test/score_recall_mean': 0.5},
 {'key': SVC(kernel=rbf), 'y/test/score_f1': [ 0.5  0.5], 'y/test/score_recall_mean/pval': [ 1.], 'y/test/score_recall/pval': [ 0.  1.], 'y/test/score_accuracy/pval': [ 1.], 'y/test/score_f1/pval': [ 1.  1.], 'y/test/score_precision/pval': [ 1.  1.], 'y/test/score_precision': [ 0.5  0.5], 'y/test/score_recall': [ 0.5  0.5], 'y/test/score_accuracy': 0.5, 'y/test/score_recall_mean': 0.5}])
```



* Run epac tree in parallel on local multi-core machine or even on HPC using [soma-workflow](https://pypi.python.org/pypi/soma-workflow "soma-workflow").

```python

    from epac import LocalEngine
    local_engine = LocalEngine(tree_root=perms_cv_svm, num_processes=2)
    perms_cv_svm = local_engine.run(X=X, y=y)
    perms_cv_svm.reduce()

```

* Design your own machine learning algorithm as a plug-in node in epac tree.

```python

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


     ## 3) run with Methods
     ## ===========================================================================
     my_svc1 = MySVC(C=1.0)
     my_svc2 = MySVC(C=2.0)
     
     two_svc = Methods(my_svc1, my_svc2)
     two_svc.reducer = MyReducer()
     
     # top-down process to call transform
     two_svc.top_down(X=X, y=y)
     # buttom-up process to compute scores
     two_svc.reduce()
     
     ## You can get below results:
     ## ===========================================================================
     ## [{'MySVC(C=1.0)': array([ 1.,  1.])}, {'MySVC(C=2.0)': array([ 1.,  1.])}]


```

Important links
---------------

**Installation**
http://neurospin.github.io/pylearn-epac/installation.html

**Tutorials**
http://neurospin.github.io/pylearn-epac/tutorials.html

**Documentation**
http://neurospin.github.io/pylearn-epac


