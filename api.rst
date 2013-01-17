# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 11:29:53 2012

@author: edouard
"""


## Use cases
## =========

# Permutation/CV SVM
# ------------------
Map(SplitKFold(SplitPermutation(X, y)), SVM())
N(SVM(), S(KFold(), S(Permutation())))



# Permutation/CV Pipeline
# -----------------------
Map(SplitKFold(SplitPermutation(X, y)), Pipeline(steps=SelectKBest(K), SVM()))

P(SVM(), N(SelectKBest(), S(KFold(), S(Permutation()))))


# Permutation/CV 2 methods
# ------------------------

Map(SplitKFold(SplitPermutation(X, y)), SplitMethods(SVM(),LDA()))
SplitMethods(SplitKFold(SplitPermutation(X, y), methods=(SVM(),LDA())
S((SVM(), LDA()), S(KFold(), S(Permutation())))


# Permutation/CV 2 nested CV
# --------------------------
Map(SplitKFold(SplitPermutation(X, y)), Pipeline(steps=[Anova()...
Map( SplitgridSearch(K=[1, 10, 100], SelectKBest(K), SVM()]))

P(S(LDA(), SVM()), P(SelectKBest(), P(GridSearch(K=[1, 10, 100]), P(Anova(), S(KFold, S(KFold, S(Permutation())))))))


Syntaxe retenue:
===============

PAR(Permutation, (args, kwargs), njobs=3),
PAR(KFold),
PAR(KFold),
N(Anova),
PAR(Grid, K=[1, 10, 100]),
N(SelectKBest) 
N(SVM, kernel="linear")

Sequential evaluation of univariate filter + SVM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
N(SelectKBest, dict(k=3)),
N(svm.SVC, dict(kernel="linear"))

10 CV of univariate filter + SVM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
PAR(KFold, dict(n=X.shape[0], n_folds=4), dict(n_jobs=5)),
    N(SelectKBest, dict(k=2)),
    N(svm.SVC, dict(kernel="linear"))

Permutation / 10CV of univariate filter + SVM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
PAR(Permutation, dict(n=100, n_perms=10), dict(n_jobs=5)),
PAR(KFold, dict(n="X.shape[0]", n_folds=3), dict(n_jobs=5))
N(SelectKBest, dict(k=3)),
N(svm.SVC, dict(kernel="linear"))

Proposition
============

Node    ::= class|class, kwargs|func, kwargs|SEQ|PAR
Splitter::= KFold|StratifiedKFold|Permutation, kwargs
SEQ     ::= SEQ(Node [, Node]*)
PAR     ::= PAR(Node [, Node]+)
        ::= PAR(Splitter, Node)

params ::= dict()
splitter_params ::= dict()
job_params ::= dict()


Examples
--------

Sequential evaluation of univariate filter + SVM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SEQ(N(SelectKBest, dict(k=3)),
    N(svm.SVC, dict(kernel="linear")))

10 CV of univariate filter + SVM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
PAR(KFold, dict(n=100, n_folds=3), dict(n_jobs=5),
    SEQ(N(SelectKBest, dict(k=3)),
        N(svm.SVC, dict(kernel="linear")))
)

Permutation / 10CV of univariate filter + SVM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
PAR(Permutation, dict(n=100, n_perms=10), dict(n_jobs=5),
    PAR(KFold, dict(n="X.shape[0]", n_folds=3), dict(n_jobs=5),
        SEQ(N(SelectKBest, dict(k=3)),
            N(svm.SVC, dict(kernel="linear")))
    )
)

10CV => Internal 3CV => Anova Ranking => Grid => SVM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
PAR(KFold, dict(n=100, n_folds=10), dict(n_jobs=10),
    PAR(KFold, dict(n="X.shape[0]", n_folds=3), dict(n_jobs=3),
        SEQ(Nfunc(f_classif, dict(output=["F","pv"])),
            PAR(Grid, dict(k=[1, 10, 100]), dict(n_jobs=3),
                SEQ(N(SelectKBest, dict()),
                    N(svm.SVC, dict(kernel="linear")))))))

