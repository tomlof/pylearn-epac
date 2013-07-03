"""
Estimator is the basic machine-learning building-bloc of the workflow.
It is a user-defined object that should implements 4 methods:


- fit(<keyword arguments>): return self.

- transform(<keyword arguments>): is called only if the estimator is a
  non-leaf node.
  Return an array or a dictionary. In the latter case, the returned dictionary
  is added to the downstream data-flow.

- predict(<keyword arguments>): is called only if the estimator is a leaf node.
  It return an array or a dictionary. In the latter the returned dictionary is
  added to results.

- score(<keyword arguments>): is called only if the estimator is a leaf node.
  It return an scalar or a dictionary. In the latter the returned dictionary is
  added to results.


@author: edouard.duchesnay@cea.fr
@author: benoit.da_mota@inria.fr
"""

## Abreviations
## tr: train
## te: test

import re
import numpy as np
from epac.workflow.base import BaseNode, key_push, key_split
from epac.utils import _func_get_args_names, train_test_merge, train_test_split, _dict_suffix_keys
from epac.utils import _sub_dict, _as_dict
from epac.map_reduce.results import ResultSet, Result
from epac.stores import StoreMem
from epac.configuration import conf
from epac.map_reduce.reducers import ClassificationReport


## ================================= ##
## == Wrapper node for estimators == ##
## ================================= ##
class Estimator(BaseNode):
    """Node that wrap estimators"""

    def get_signature(self):
        """Overload the base name method"""
        if not self.signature_args:
            return self.estimator.__class__.__name__
        else:
            args_str = ",".join([str(k) + "=" + str(self.signature_args[k])
                             for k in self.signature_args])
            args_str = "(" + args_str + ")"
            return self.estimator.__class__.__name__ + args_str

    def get_parameters(self):
        return self.estimator.__dict__


class InternalEstimator(Estimator):
    """Estimator Wrapper: Automatically connect estimator.fit and
    estimator.transform to BaseNode.transform.

    Parameters:
        estimator: object that implement fit and transform

    Automatically connect estimator.fit (if exist) and estimator.transform

    Example
    -------
    >>> from sklearn import datasets
    >>> from sklearn.svm import SVC
    >>> from sklearn.lda import LDA
    >>> from sklearn.feature_selection import SelectKBest
    >>> from epac.workflow.estimators import InternalEstimator
    >>>
    >>> X, y = datasets.make_classification(n_samples=12,
    ...                                     n_features=10,
    ...                                     n_informative=2,
    ...                                     random_state=1)
    >>> Xy = dict(X=X, y=y)
    >>> internal_estimator  = InternalEstimator(SelectKBest(k=2))
    >>> internal_estimator.transform(**Xy)
    {'y': array([1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1]), 'X': array([[-0.34385368,  0.75623409],
           [ 0.19829972, -1.16389861],
           [-0.74715829,  0.86977629],
           [ 1.13162939,  0.90876519],
           [ 0.23009474, -0.68017257],
           [ 0.16003707, -1.55458039],
           [ 0.40349164,  1.38791468],
           [-1.11731035,  0.23476552],
           [ 1.19891788,  0.0888684 ],
           [-0.75439794, -0.90039992],
           [ 0.12015895,  2.05996541],
           [-0.20889423,  2.05313908]])}
    """
    def __init__(self, estimator, in_args_fit=None, in_args_transform=None):
        """
        Parameters
        ----------
        estimator: any class contains fit and transform functions
            any class implements fit and transform

        in_args_fit: list of strings
            names of input arguments of the fit method. If missing discover
            discover it automatically.

        in_args_transform: list of strings
            names of input arguments of the tranform method. If missing,
            discover it automatically.
        """
        if not hasattr(estimator, "fit") or not \
            hasattr(estimator, "transform"):
            raise ValueError("estimator should implement fit and transform")
        super(InternalEstimator, self).__init__()
        self.estimator = estimator
        self.in_args_fit = _func_get_args_names(self.estimator.fit) \
            if in_args_fit is None else in_args_fit
        self.in_args_transform = _func_get_args_names(self.estimator.transform) \
            if in_args_transform is None else in_args_transform

    def transform(self, **Xy):
        """
        Parameter
        ---------
        Xy: dictionary
            parameters for fit and transform
        """
        if conf.KW_SPLIT_TRAIN_TEST in Xy:
            Xy_train, Xy_test = train_test_split(Xy)
            self.estimator.fit(**_sub_dict(Xy_train, self.in_args_fit))
            # catch args_transform in ds, transform, store output in a dict
            Xy_out_tr = _as_dict(self.estimator.transform(
                        **_sub_dict(Xy_train,self.in_args_transform)),
                        keys=self.in_args_transform)
            Xy_out_te = _as_dict(self.estimator.transform(**_sub_dict(Xy_test,
                            self.in_args_transform)),
                            keys=self.in_args_transform)
            Xy_out = train_test_merge(Xy_out_tr, Xy_out_te)
        else:
            self.estimator.fit(**_sub_dict(Xy, self.in_args_fit))
            # catch args_transform in ds, transform, store output in a dict
            Xy_out = _as_dict(self.estimator.transform(**_sub_dict(Xy,
                                                 self.in_args_transform)),
                           keys=self.in_args_transform)
        # update ds with transformed values
        Xy.update(Xy_out)
        return Xy


    def reduce(self, store_results=True):
        # 1) Build sub-aggregates over children
        children_result_set = [child.reduce(store_results=False) for
            child in self.children]
        result_set = ResultSet(*children_result_set)
        # Append node signature in the keys
        for result in result_set:
            result["key"] = key_push(self.get_signature(), result["key"])
        return result_set


class LeafEstimator(Estimator):
    """Estimator Wrapper: 
    Automatically connect estimator.fit (if exist) and estimator.predict to 
    BaseNode.transform.

    Example
    -------
    >>> from sklearn import datasets
    >>> from sklearn.svm import SVC
    >>> from sklearn.lda import LDA
    >>> from sklearn.feature_selection import SelectKBest
    >>> from epac.workflow.estimators import LeafEstimator
    >>>
    >>> X, y = datasets.make_classification(n_samples=12,
    ...                                     n_features=10,
    ...                                     n_informative=2,
    ...                                     random_state=1)
    >>> Xy = dict(X=X, y=y)
    >>> leaf_estimator  = LeafEstimator(SVC())
    >>> leaf_estimator.transform(**Xy)
    {'y': array([1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1])}

    """
    def __init__(self, estimator, in_args_fit=None, in_args_predict=None,
                 out_args_predict=None):
        '''
        Parameters
        ----------
        estimator: any class contains fit and predict functions 
            any class implements fit and predict

        in_args_fit: list of strings
            names of input arguments of the fit method. If missing discover
            discover it automatically.

        in_args_predict: list of strings
            names of input arguments of the predict method. If missing,
            discover it automatically.

        out_args_predict: list of strings
            names of output arguments of the predict method. If missing,
            discover it automatically by self.in_args_fit - in_args_predict.
            If not differences (such with PCA with fit(X) and predict(X))
            use in_args_predict.
        '''
        if not hasattr(estimator, "fit") or not \
            hasattr(estimator, "predict"):
            raise ValueError("estimator should implement fit and predict")
        super(LeafEstimator, self).__init__()
        self.estimator = estimator
        self.in_args_fit = _func_get_args_names(self.estimator.fit) \
            if in_args_fit is None else in_args_fit
        self.in_args_predict = _func_get_args_names(self.estimator.predict) \
            if in_args_predict is None else in_args_predict
        if out_args_predict is None:
            fit_predict_diff = list(set(self.in_args_fit).difference(self.in_args_predict))
            if len(fit_predict_diff) > 0:
                self.out_args_predict = fit_predict_diff
            else:
                self.out_args_predict = self.in_args_predict
        else:
            self.out_args_predict =  out_args_predict

    """Extimator Wrapper: connect fit + predict to transform"""
    def transform(self, **Xy):
        """
        Parameter
        ---------
        Xy: dictionary
            parameters for fit and transform
        """
        if conf.KW_SPLIT_TRAIN_TEST in Xy:
            Xy_train, Xy_test = train_test_split(Xy)
            Xy_out = dict()
            # Train fit
            self.estimator.fit(**_sub_dict(Xy_train, self.in_args_fit))
            # Train predict
            Xy_out_tr = _as_dict(self.estimator.predict(**_sub_dict(Xy_train,
                                                 self.in_args_predict)),
                           keys=self.out_args_predict)
            Xy_out_tr = _dict_suffix_keys(Xy_out_tr,
                suffix=conf.SEP + conf.TRAIN + conf.SEP + conf.PREDICTION)
            Xy_out.update(Xy_out_tr)
            # Test predict
            Xy_out_te = _as_dict(self.estimator.predict(**_sub_dict(Xy_test,
                                                 self.in_args_predict)),
                           keys=self.out_args_predict)
            Xy_out_te = _dict_suffix_keys(Xy_out_te,
                suffix=conf.SEP + conf.TEST + conf.SEP + conf.PREDICTION)
            Xy_out.update(Xy_out_te)
            ## True test
            Xy_test_true = _sub_dict(Xy_test, self.out_args_predict)
            Xy_out_true = _dict_suffix_keys(Xy_test_true,
                suffix=conf.SEP + conf.TEST + conf.SEP + conf.TRUE)
            Xy_out.update(Xy_out_true)
        else:
            self.estimator.fit(**_sub_dict(Xy, self.in_args_fit))
            # catch args_transform in ds, transform, store output in a dict
            Xy_out = _as_dict(self.estimator.predict(**_sub_dict(Xy,
                                                 self.in_args_predict)),
                           keys=self.out_args_predict)
        return Xy_out

    def reduce(self, store_results=True):
        return self.load_state(name="result_set")


class CVBestSearchRefit(Estimator):
    """Cross-validation + grid-search then refit with optimals parameters.

    Average results over first axis, then find the arguments that maximize or
    minimise a "score" over other axis.

    Parameters
    ----------

    See CV parameters, plus other parameters:

    score: string
        the score name to be optimized (default "mean_score_te").

    arg_max: boolean
        True/False take parameters that maximize/minimize the score. Default
        is True.
    """

    def __init__(self, node, **kwargs):
        super(CVBestSearchRefit, self).__init__(estimator=None)
        score = kwargs.pop("score") if "score" in kwargs else "mean_score_te"
        arg_max = kwargs.pop("arg_max") if "arg_max" in kwargs else True
        from epac.workflow.splitters import CV
        #methods = Methods(*tasks)
        cv = CV(node=node, reducer=ClassificationReport(keep=False), **kwargs)
        self.score = score
        self.arg_max = arg_max
        self.add_child(cv)  # first child is the CV

    def get_signature(self):
        return self.__class__.__name__

    def get_children_top_down(self):
        """Return children during the top-down execution."""
        return []

    def transform(self, **Xy):
        Xy_train, Xy_test = train_test_split(Xy)
        self._fit(**Xy_train)
        Xy_test = self._predict(**Xy_test)
        return Xy_test

    def _fit(self, **Xy):
        # Fit/predict CV grid search
        cv = self.children[0]
        cv.store = StoreMem()  # local store erased at each fit
        from epac.workflow.splitters import CV
        from epac.workflow.pipeline import Pipe
        if not isinstance(cv, CV):
            raise ValueError('Child of %s is not a "CV."'
            % self.__class__.__name__)
        cv.top_down(**Xy)
        #  Pump-up results
        cv_result_set = cv.reduce(store_results=False)
        key_val = [(result.key(), result[self.score]) for result in cv_result_set]
        mean_cv = np.asarray(zip(*key_val)[1])
        mean_cv_opt = np.max(mean_cv) if self.arg_max else  np.min(mean_cv)
        idx_best = np.where(mean_cv == mean_cv_opt)[0][0]
        best_key = key_val[idx_best][0]
        # Find nodes that match the best
        nodes_dict = {n.get_signature(): n for n in self.walk_true_nodes() \
            if n.get_signature() in key_split(best_key)}
        refited = Pipe(*[nodes_dict[k].estimator for k in key_split(best_key)])
        refited.store = StoreMem()    # local store erased at each fit
        self.children = self.children[:1]
        self.add_child(refited)
        refited.fit(recursion=True, **Xy)
        refited_result_set = refited.reduce(store_results=False)
        result_set = ResultSet(refited_result_set)
        result = result_set.values()[0]  # There is only one
        result["key"] = self.get_signature()
        result["best_params"] = [dict(sig) for sig in key_split(best_key, eval=True)]
        self.save_state(result_set, name="result_set")
        #to_refit.bottum_up(store_results=False)
        # Delete (eventual) about previous refit
        return self

    def _predict(self, recursion=True, **Xy):
        """Call transform  with sample_set="test" """
        refited = self.children[1]
        pred = refited.top_down(**Xy)
        # Update current results with refited prediction
        refited_result = refited.reduce(store_results=False).values()[0]
        result = self.load_state(name="result_set").values()[0]
        result.update(refited_result.payload())
        return pred



    def reduce(self, store_results=True):
        # Terminaison (leaf) node return result_set
        return self.load_state(name="result_set")

if __name__ == "__main__":
    import doctest
    doctest.testmod()
