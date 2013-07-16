.. _examples:

========
Examples
========


In this section, we are going to show several small examples to understand epac.
In order to build the epac tree, we first need to build X matrix (samples) 
and y vector (labels) for machine learning.

::

    from sklearn import datasets
    from sklearn.svm import LinearSVC as SVM
    from sklearn.lda import LDA
    from sklearn.feature_selection import SelectKBest
    X, y = datasets.make_classification(n_samples=12, 
                                        n_features=10,
                                        n_informative=2)


In epac, there are two very basic elements, *Pipe* (sequential pipeline) and *Methods* (parallel methods). 
*Pipe* is used for running a sequence of nodes while *Method* is designed for parallelly run sequences of nodes. 
 

Now, we first introduce *Pipe*, which executes a sequential of BaseNode. These nodes should implement 

either as non-leaf node,

- fit and transform, e.g. *SelectKBest(k=2)*

or as leaf node

- fit and predict, e.g. *SVM()*

::

    # Build sequential Pipeline
    # -------------------------
    # 2  SelectKBest (Estimator)
    # |
    # SVM Classifier (Estimator)
    from epac import Pipe
    pipe = Pipe(SelectKBest(k=2), SVM())
    pipe.run(X=X, y=y)
     
    # The downstream data-flow is a keyword arguments (dict) containing X and y.
    # It will pass through each processing node, SelectKBest(k=2) and SVM.
    # Each node call the "transform" method, that take a dictionnary as input
    # and produces a dictionnary as output. The output is passed  to the next node. 
    
    # The return value of the run is simply agregation of the outputs (dict) of
    # the leaf nodes


::

    ## Parallelization
    ## ===============
    
    # Multi-classifiers
    # -----------------
    #         Methods       Methods (Splitter)
    #        /   \
    # SVM(C=1)  SVM(C=10)   Classifiers (Estimator)
    from epac import Methods
    multi = Methods(SVM(C=1), SVM(C=10))
    multi.run(X=X, y=y)
    print multi.reduce()

    # Reduce format outputs into "ResultSet" which is a dict-like structure
    # which contains the "keys" of the methods that as beeen used.

In epac, **run** means the top-down process, and **reduce** means bottom-up process.
All the input is defined as dictionary, for example "X=X, y=y" in pipe.run(X=X, y=y).     
  
::    
    
    #                         Methods                  Methods (Splitter)
    #          /                        \
    # SVM(l1, C=1)  SVM(l1, C=10)  ..... SVM(l2, C=10) Classifiers (Estimator)
    svms = Methods(*[SVM(loss=loss, C=C) for loss in ("l1", "l2") for C in [1, 10]])
    svms.run(X=X, y=y)
    print svms.reduce()
    
    # Parallelize sequential Pipeline: Anova(k best selection) + SVM.
    #    Methods    Methods (Splitter)
    #  /   |   \
    # 1    5   10   SelectKBest (Estimator)
    # |    |    |
    # SVM SVM SVM   Classifiers (Estimator)
    anovas_svm = Methods(*[Pipe(SelectKBest(k=k), SVM()) for k in [1, 2]])
    anovas_svm.run(X=X, y=y)
    print anovas_svm.reduce()
    
    
    # Cross-validation
    # ----------------
    # CV of LDA
    #      CV                 (Splitter)
    #  /   |   \
    # 0    1    2  Folds      (Slicer)
    # |    |
    #   Methods               (Splitter)
    #    /   \
    #  LDA  SVM    Classifier (Estimator)
    from epac import CV, Methods
    cv = CV(Methods(LDA(), SVM()))
    cv.run(X=X, y=y)
    print cv.reduce()
    
    
    # Model selection using CV
    # ------------------------
    # CVBestSearchRefit
    #      Methods       (Splitter)
    #      /    \
    # SVM(C=1)  SVM(C=10)   Classifier (Estimator)
    from epac import Pipe, CVBestSearchRefit, Methods
    # CV + Grid search of a simple classifier
    wf = CVBestSearchRefit(Methods(SVM(C=1), SVM(C=10)))
    wf.run(X=X, y=y)
    print wf.reduce()
    
    # Feature selection combined with SVM and LDA
    # CVBestSearchRefit
    #                     Methods          (Splitter)
    #               /              \
    #            KBest(1)         KBest(5) SelectKBest (Estimator)
    #              |
    #            Methods                   (Splitter)
    #        /          \
    #    LDA()          SVM() ...          Classifiers (Estimator)
    pipelines = Methods(*[Pipe(SelectKBest(k=k), Methods(LDA(), SVM())) for k in [1, 5]])
    print [n for n in pipelines.walk_leaves()]
    best_cv = CVBestSearchRefit(pipelines)
    best_cv.run(X=X, y=y)
    best_cv.reduce()
    
    # Put it in an outer CV
    cv = CV(best_cv)
    cv.run(X=X, y=y)
    cv.reduce()
    
    # Perms + Cross-validation of SVM(linear) and SVM(rbf)
    # -------------------------------------
    #           Perms        Perm (Splitter)
    #      /     |       \
    #     0      1       2   Samples (Slicer)
    #            |
    #           CV           CV (Splitter)
    #       /   |   \
    #      0    1    2       Folds (Slicer)
    #           |
    #        Methods         Methods (Splitter)
    #    /           \
    # SVM(linear)  SVM(rbf)  Classifiers (Estimator)
    
    from sklearn.svm import SVC
    from epac import Perms, CV, Methods
    perms_cv_svm = Perms(CV(Methods(*[SVC(kernel="linear"), SVC(kernel="rbf")])))
    perms_cv_svm.run(X=X, y=y)
    perms_cv_svm.reduce()
    
    
    # Run with soma-workflow for multi-processes
    from epac import SomaWorkflowEngine
    sfw_engine = SomaWorkflowEngine(
                        tree_root=perms_cv_svm,
                        num_processes=2,
                        )
    perms_cv_svm = sfw_engine.run(X=X, y=y)
    perms_cv_svm.reduce()
