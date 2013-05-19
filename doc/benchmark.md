=================================
Performances and memory footprint
=================================

Workflow
--------

Permutation / CV / ParCVGridSearchRefit of SelectKbest(k_values) / SVM(C_values)

No memory usage reducing, ParCVGridSearchRefit store nested CV
--------------------------------------------------------------

Command::

```
    python examples/run_single_process.py --n_folds=10 --n_folds_nested=5 --n_features=10000 --k_max=100
    k_values = [1, 2, 4, 8, 16, 32, 64, 100]
    C_values = [1, 10]
```

```
    Step              | Time  s/permutation | Memory MB/permutation | 100perms
    Tree construction | 0.12s               | 3-6                   | 12s, 500MB
    Fit_predict       | 15s                 | 68-80                 | 25mns, 6GB
    Reduce            | 0.000469439s        |                       |
```
After nocopy, ParCVGridSearchRefit store nested CV
--------------------------------------------------

Command::

```
    python examples/run_single_process.py --n_folds=10 --n_folds_nested=5 --n_features=10000 --k_max=100
    k_values = [1, 2, 4, 8, 16, 32, 64, 100]
    C_values = [1, 10]
```

```
    Step              | Time  s/permutation | Memory MB/permutation | 100perms
    Tree construction | 0.000041s           | 0.62                  | 0.0041s, 62MB
    Fit_predict       | 14s                 | 0.65                  | 23.8mns, 65MB
    Reduce            | 0.0038995s          |                       | 0.39
```



