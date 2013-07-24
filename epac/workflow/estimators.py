"""
Estimator wrap ML procedure into EPAC Node. To be EPAC compatible, one should
inherit from BaseNode and implement the "transform" method.

InternalEstimator and LeafEstimator aim to provide automatic wrapper to objects
that implement fit and predict methods.

@author: edouard.duchesnay@cea.fr
@author: jinpeng.li@cea.fr
"""

## Abreviations
## tr: train
## te: test

import numpy as np
from epac.workflow.base import BaseNode, key_push, key_split
from epac.utils import _func_get_args_names, train_test_merge, train_test_split, _dict_suffix_keys
from epac.utils import _sub_dict, _as_dict
from epac.map_reduce.results import ResultSet, Result
from epac.stores import StoreMem
from epac.configuration import conf
from epac.map_reduce.reducers import ClassificationReport
from epac.workflow.wrappers import Wrapper


class InternalEstimator(Wrapper):
    """Estimator Wrapper: Automatically connect wrapped_node.fit and
    wrapped_node.transform to BaseNode.transform.

    Parameters:
        wrapped_node: object that implement fit and transform

    Example
    -------
    >>> from sklearn import datasets
    >>> from sklearn.svm import SVC
    >>> from sklearn.lda import LDA
    >>> from sklearn.feature_selection import SelectKBest
    >>> from epac.workflow.estimators import InternalEstimator
    >>> X, y = datasets.make_classification(n_samples=12,
    ...                                     n_features=10,
    ...                                     n_informative=2,
    ...                                     random_state=1)
    >>> Xy = dict(X=X, y=y)
    >>> print Xy
    {'y': array([ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  1.,  0.,  1.]), 'X': array([[ 1.2245077 , -0.44712856, -0.65286929, -0.34385368,  0.01512057,
            -0.04831591,  0.75623409,  0.04359686,  0.69803203, -0.62000084],
           [ 1.12948391,  0.12182127,  2.21822522,  0.19829972,  1.46919607,
            -2.51338666, -1.16389861,  0.11900865,  0.37756379, -0.67066229],
           [ 2.10025514,  0.19091548, -2.02608528, -0.74715829, -1.55106565,
             2.66393293,  0.86977629,  1.6924546 , -0.63699565,  0.05080775],
           [-0.50446586, -1.44411381, -2.10890879,  1.13162939, -1.61075629,
             2.76628954,  0.90876519,  1.51981682, -1.39649634,  2.18557541],
           [ 0.41005165,  0.18656139,  1.33239455,  0.23009474,  0.90296627,
            -1.54575204, -0.68017257,  0.76201118, -0.20075807, -0.22232814],
           [ 0.82797464, -0.30620401,  1.49115503,  0.16003707,  0.15225736,
            -0.21856666, -1.55458039,  0.87616892, -2.02220122,  0.31563495],
           [-0.9537006 ,  0.74055645, -0.58514508,  0.40349164,  0.78180501,
            -1.39610781,  1.38791468,  0.59357852,  0.16938243, -1.09491185],
           [-0.88762896, -0.19183555, -1.30057302, -1.11731035, -1.34569135,
             2.32639651,  0.23476552,  0.2344157 ,  0.74204416,  1.65980218],
           [ 0.07734007,  0.42349435,  0.26063124,  1.19891788,  0.41671258,
            -0.7251233 ,  0.0888684 ,  0.18515642, -0.63873041, -0.37528495],
           [-0.07557171,  0.48851815,  0.29975053, -0.75439794, -0.60541421,
             1.07602248, -0.90039992,  1.25286816, -0.29809284,  0.51292982],
           [-0.34934272, -1.1425182 , -2.16279883,  0.12015895, -0.43161072,
             0.68816127,  2.05996541,  0.61720311, -0.35224985,  0.30017032],
           [ 0.88514116,  0.28558733, -3.62072922, -0.20889423, -2.23221458,
             3.81037776,  2.05313908,  0.58662319,  0.93110208,  0.83898341]])}
    >>> internal_estimator  = InternalEstimator(SelectKBest(k=2))
    >>> internal_estimator.transform(**Xy)
    {'y': array([ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  1.,  0.,  1.]), 'X': array([[-0.34385368,  0.75623409],
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
    def __init__(self, wrapped_node, in_args_fit=None, in_args_transform=None):
        """
        Parameters
        ----------
        wrapped_node: any class contains fit and transform functions
            any class implements fit and transform

        in_args_fit: list of strings
            names of input arguments of the fit method. If missing discover
            discover it automatically.

        in_args_transform: list of strings
            names of input arguments of the tranform method. If missing,
            discover it automatically.
        """
        if not hasattr(wrapped_node, "fit") or not \
            hasattr(wrapped_node, "transform"):
            raise ValueError("wrapped_node should implement fit and transform")
        super(InternalEstimator, self).__init__(wrapped_node=wrapped_node)
        self.in_args_fit = _func_get_args_names(self.wrapped_node.fit) \
            if in_args_fit is None else in_args_fit
        self.in_args_transform = \
            _func_get_args_names(self.wrapped_node.transform) \
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
            res = self.wrapped_node.fit(**_sub_dict(Xy_train, self.in_args_fit))
            # catch args_transform in ds, transform, store output in a dict
            Xy_out_tr = _as_dict(self.wrapped_node.transform(
                        **_sub_dict(Xy_train, self.in_args_transform)),
                        keys=self.in_args_transform)
            Xy_out_te = _as_dict(self.wrapped_node.transform(**_sub_dict(Xy_test,
                            self.in_args_transform)),
                            keys=self.in_args_transform)
            Xy_out = train_test_merge(Xy_out_tr, Xy_out_te)
        else:
            res = self.wrapped_node.fit(**_sub_dict(Xy, self.in_args_fit))
            # catch args_transform in ds, transform, store output in a dict
            Xy_out = _as_dict(self.wrapped_node.transform(**_sub_dict(Xy,
                                                 self.in_args_transform)),
                           keys=self.in_args_transform)
        # update ds with transformed values
        Xy.update(Xy_out)
        return Xy


class LeafEstimator(Wrapper):
    """Estimator Wrapper:
    Automatically connect wrapped_node.fit and
    wrapped_node.predict to BaseNode.transform.

    Example
    -------
    >>> from sklearn import datasets
    >>> from sklearn.svm import SVC
    >>> from sklearn.lda import LDA
    >>> from sklearn.feature_selection import SelectKBest
    >>> from epac.workflow.estimators import LeafEstimator
    >>> X, y = datasets.make_classification(n_samples=12,
    ...                                     n_features=10,
    ...                                     n_informative=2,
    ...                                     random_state=1)
    >>> Xy = dict(X=X, y=y)
    >>> leaf_estimator  = LeafEstimator(SVC())
    >>> leaf_estimator.transform(**Xy)
    {'y/true': array([ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  1.,  0.,  1.]), 'y/pred': array([ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  1.,  0.,  1.])}
    >>> print leaf_estimator.reduce()
    None
    >>> leaf_estimator.top_down(**Xy)
    {'y/true': array([ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  1.,  0.,  1.]), 'y/pred': array([ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  1.,  0.,  1.])}
    >>> print leaf_estimator.reduce()
    ResultSet(
    [{'key': SVC, 'y/true': [ 1.  0.  0.  1.  0.  0.  1.  0.  1.  1.  0.  1.], 'y/pred': [ 1.  0.  0.  1.  0.  0.  1.  0.  1.  1.  0.  1.]}])

    """
    def __init__(self, wrapped_node, in_args_fit=None, in_args_predict=None,
                 out_args_predict=None):
        '''
        Parameters
        ----------
        wrapped_node: any class contains fit and predict functions 
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
        if not hasattr(wrapped_node, "fit") or not \
            hasattr(wrapped_node, "predict"):
            raise ValueError("%s should implement fit and predict" %
                              wrapped_node.__class__.__name__)
        super(LeafEstimator, self).__init__(wrapped_node=wrapped_node)
        self.in_args_fit = _func_get_args_names(self.wrapped_node.fit) \
            if in_args_fit is None else in_args_fit
        self.in_args_predict = _func_get_args_names(self.wrapped_node.predict)\
            if in_args_predict is None else in_args_predict
        if out_args_predict is None:
            fit_predict_diff = list(set(self.in_args_fit).difference(
                                        self.in_args_predict))
            if len(fit_predict_diff) > 0:
                self.out_args_predict = fit_predict_diff
            else:
                self.out_args_predict = self.in_args_predict
        else:
            self.out_args_predict = out_args_predict

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
            res = self.wrapped_node.fit(**_sub_dict(Xy_train, self.in_args_fit))
            # Train predict
            Xy_out_tr = _as_dict(self.wrapped_node.predict(**_sub_dict(Xy_train,
                                                 self.in_args_predict)),
                           keys=self.out_args_predict)
            Xy_out_tr = _dict_suffix_keys(Xy_out_tr,
                suffix=conf.SEP + conf.TRAIN + conf.SEP + conf.PREDICTION)
            Xy_out.update(Xy_out_tr)
            # Test predict
            Xy_out_te = _as_dict(self.wrapped_node.predict(**_sub_dict(Xy_test,
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
            res = self.wrapped_node.fit(**_sub_dict(Xy, self.in_args_fit))
            # catch args_transform in ds, transform, store output in a dict
            Xy_out = _as_dict(self.wrapped_node.predict(**_sub_dict(Xy,
                                                 self.in_args_predict)),
                           keys=self.out_args_predict)
            Xy_out = _dict_suffix_keys(Xy_out,
                suffix=conf.SEP + conf.PREDICTION)
            ## True test
            Xy_true = _sub_dict(Xy, self.out_args_predict)
            Xy_out_true = _dict_suffix_keys(Xy_true,
                suffix=conf.SEP + conf.TRUE)
            Xy_out.update(Xy_out_true)
        return Xy_out


class CVBestSearchRefit(Wrapper):
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

    Example
    -------
    >>> from sklearn import datasets
    >>> from sklearn.svm import SVC
    >>> from epac import Methods
    >>> from epac.workflow.estimators import CVBestSearchRefit
    >>> X, y = datasets.make_classification(n_samples=12,
    ... n_features=10,
    ... n_informative=2,
    ... random_state=1)
    >>> n_folds_nested = 2
    >>> C_values = [.1, 0.5, 1, 2, 5]
    >>> kernels = ["linear", "rbf"]
    >>> methods = Methods(*[SVC(C=C, kernel=kernel)
    ...     for C in C_values for kernel in kernels])
    >>> wf = CVBestSearchRefit(methods, n_folds=n_folds_nested)
    >>> wf.transform(X=X, y=y)
    {'best_params': [{'kernel': 'linear', 'C': 2, 'name': 'SVC'}], 'y/true': array([ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  1.,  0.,  1.]), 'y/pred': array([ 0.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,  1.])}
    >>> wf.reduce()
    >>> wf.run(X=X, y=y)
    {'best_params': [{'kernel': 'linear', 'C': 2, 'name': 'SVC'}], 'y/true': array([ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  1.,  0.,  1.]), 'y/pred': array([ 0.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,  1.])}
    >>> wf.reduce()
    ResultSet(
    [{'key': CVBestSearchRefit, 'best_params': [{'kernel': 'linear', 'C': 2, 'name': 'SVC'}], 'y/true': [ 1.  0.  0.  1.  0.  0.  1.  0.  1.  1.  0.  1.], 'y/pred': [ 0.  0.  0.  1.  0.  0.  1.  0.  1.  0.  0.  1.]}])

    """

    def __init__(self, node, **kwargs):
        super(CVBestSearchRefit, self).__init__(wrapped_node=None)
        #### 'y/test/score_recall_mean'
        default_score = "y" + conf.SEP + \
                        conf.TEST + conf.SEP + \
                        conf.SCORE_RECALL_MEAN
        score = kwargs.pop("score") if "score" in kwargs else default_score
        arg_max = kwargs.pop("arg_max") if "arg_max" in kwargs else True
        from epac.workflow.splitters import CV
        #methods = Methods(*tasks)
        self.cv = CV(node=node, reducer=ClassificationReport(keep=False), **kwargs)
        self.score = score
        self.arg_max = arg_max

    def get_signature(self):
        return self.__class__.__name__

    def transform(self, **Xy):
        Xy_train, Xy_test = train_test_split(Xy)
        if Xy_train is Xy_test:
            to_refit, best_params = self._search_best(**Xy)
        else:
            to_refit, best_params = self._search_best(**Xy_train)
        out = to_refit.top_down(**Xy)
        out[conf.BEST_PARAMS] = best_params
        self.refited = to_refit
        self.best_params = best_params
        return out

    def _search_best(self, **Xy):
        # Fit/predict CV grid search
        self.cv.store = StoreMem()  # local store erased at each fit
        from epac.workflow.pipeline import Pipe
        self.cv.top_down(**Xy)
        #  Pump-up results
        cv_result_set = self.cv.reduce(store_results=False)
        key_val = [(result.key(), result[self.score]) \
                for result in cv_result_set]
        scores = np.asarray(zip(*key_val)[1])
        scores_opt = np.max(scores) if self.arg_max else  np.min(scores)
        idx_best = np.where(scores == scores_opt)[0][0]
        best_key = key_val[idx_best][0]
        # Find nodes that match the best
        nodes_dict = {n.get_signature(): n for n in self.cv.walk_true_nodes() \
            if n.get_signature() in key_split(best_key)}
        to_refit = Pipe(*[nodes_dict[k].wrapped_node for k in key_split(best_key)])
        best_params = [dict(sig) for sig in key_split(best_key, eval=True)]
        return to_refit, best_params

    def reduce(self, store_results=True):
        # Terminaison (leaf) node return result_set
        return self.load_state(name=conf.RESULT_SET)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
