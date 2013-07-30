# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 14:59:37 2013


Still working on. It is not finished.

@author: jinpeng.li@cea.fr

"""


from sklearn import datasets
from sklearn.svm import SVC
from epac.map_reduce.reducers import Reducer
from epac import Methods
from sklearn.metrics import precision_recall_fscore_support
from sklearn.cross_validation import StratifiedKFold
from epac.workflow.splitters import BaseNodeSplitter
from epac.workflow.base import BaseNode
from epac import Pipe
from epac.workflow.factory import NodeFactory
import copy


## 1) Build dataset
## ===========================================================================
X, y = datasets.make_classification(n_samples=12,
                                    n_features=10,
                                    n_informative=2,
                                    random_state=1)


## 2) Design your classifier
## ===========================================================================
class MyCVSVC:
    def __init__(self, C=1.0):
        self.C = C
    def transform(self, X_train, y_train, X_test, y_test):
        svc = SVC(C=self.C)
        svc.fit(X_train, y_train)
        y_train_pred = svc.predict(X_train)
        y_test_pred = svc.predict(X_test)
        return {"y/train/pred": y_train_pred, "y/train/true": y_test,
                "y/test/pred": y_test_pred, "y/test/true": y_test}


class MyTrainTestCV(BaseNode):
    def __init__(self, i_flods=0, train_index=0, test_index=0):
        super(MyTrainTestCV, self).__init__()
        self.train_index = train_index
        self.test_index = test_index
        self.i_flods = i_flods
        self.signature_args = repr(i_flods)
        self.signature_name = "CV"
    def transform(self, X, y):
        X_train = X[self.train_index]
        y_train = y[self.train_index]
        X_test = X[self.test_index]
        y_test = y[self.test_index]
        return dict(X_train=X_train, y_train=y_train,
                    X_test=X_test, y_test=y_test)
    def get_parameters(self):
        return dict(i_flods=self.i_flods)
    def get_signature(self):
        """Overload the base name method: use self.signature_name"""
        return self.signature_name + \
            "(nb=" + str(self.signature_args) + ")"
    def get_signature_args(self):
        """overried get_signature_args to return a copy"""
        return copy.copy(self.signature_args)


class MyCV(BaseNodeSplitter):
    def __init__(self, node, y, n_flods=2):
        super(MyCV, self).__init__()
        node = NodeFactory.build(node)
        skf = StratifiedKFold(y, n_flods)
        i_flods = 0
        for train_index, test_index in skf:
            cp_node = copy.copy(node)
            cp_node.__init__(i_flods, train_index, test_index)
            self.add_child(cp_node)
            i_flods = i_flods + 1
    def transform(self, X, y):
        return dict(X=X, y=y)


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

## 4) run with Methods
## ===========================================================================
my_svc1 = MyCVSVC(C=1.0)
my_svc2 = MyCVSVC(C=2.0)

two_svc = Pipe(MyTrainTestCV(), Methods(my_svc1, my_svc2))
cv_two_svc = MyCV(two_svc, y, n_flods=2)

for leaf in cv_two_svc.walk_leaves():
    print leaf

# top-down process to call transform
cv_two_svc.top_down(X=X, y=y)
# buttom-up process to compute scores
cv_two_svc.reduce()


## You can get below results:
## ===========================================================================
## [{'MySVC(C=1.0)': array([ 1.,  1.])}, {'MySVC(C=2.0)': array([ 1.,  1.])}]
