# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 17:05:28 2013

@author: jinpeng.li@cea.fr

"""

from sklearn import datasets
from sklearn.svm import SVC
from epac.map_reduce.reducers import Reducer
from epac import Methods
from sklearn.metrics import precision_recall_fscore_support


## 1) Build dataset
## ===========================================================================
X, y = datasets.make_classification(n_samples=12,
                                    n_features=10,
                                    n_informative=2,
                                    random_state=1)


## 2) Design your classifier
## ===========================================================================
class MySVC:
    def __init__(self, C=1.0):
        self.C = C

    def transform(self, X, y):
        svc = SVC(C=self.C)
        svc.fit(X, y)
        # "transform" should return a dictionary
        return {"y/pred": svc.predict(X)}


## 3) Design your reducer which compute, precision, recall, f1_score, etc.
## ===========================================================================
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

## 4) Design your reducer
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
