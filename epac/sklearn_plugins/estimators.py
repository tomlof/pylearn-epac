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

from epac.utils import _func_get_args_names
from epac.utils import train_test_merge
from epac.utils import train_test_split
from epac.utils import _dict_suffix_keys
from epac.utils import _sub_dict, _as_dict
from epac.configuration import conf
from epac.workflow.wrappers import Wrapper


class Estimator(Wrapper):
    """Estimator Wrapper: Automatically connect wrapped_node.fit and
    wrapped_node.transform to BaseNode.transform

    Example
    -------
    >>> from sklearn.lda import LDA
    >>> from sklearn import datasets
    >>> from sklearn.feature_selection import SelectKBest
    >>> from sklearn.svm import SVC
    >>> from epac import Pipe
    >>> from epac import CV, Methods
    >>> from epac.sklearn_plugins import Estimator
    >>>
    >>> X, y = datasets.make_classification(n_samples=15,
    ...                                     n_features=10,
    ...                                     n_informative=7,
    ...                                     random_state=5)
    >>> Xy = dict(X=X, y=y)
    >>> lda_estimator = Estimator(LDA())
    >>> lda_estimator.transform(**Xy)
    {'y/true': array([ 1.,  0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,
            1.,  1.]), 'y/pred': array([ 1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,
            1.,  1.])}
    >>> pipe = Pipe(SelectKBest(k=7), lda_estimator)
    >>> pipe.run(**Xy)
    {'y/true': array([ 1.,  0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,
            1.,  1.]), 'y/pred': array([ 1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,
            1.,  1.])}
    >>> pipe2 = Pipe(lda_estimator, SVC())
    >>> pipe2.run(**Xy)
    {'y/true': array([ 1.,  0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,
            1.,  1.]), 'y/pred': array([ 1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,
            1.,  1.])}
    >>> cv = CV(Methods(pipe, SVC()), n_folds=3)
    >>> cv.run(**Xy)
    [[{'y/test/pred': array([ 1.,  0.,  1.,  1.,  1.]), 'y/train/pred': array([ 1.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,  1.]), 'y/test/true': array([ 0.,  1.,  0.,  0.,  1.])}, {'y/test/pred': array([ 0.,  0.,  0.,  1.,  1.]), 'y/train/pred': array([ 1.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,  1.]), 'y/test/true': array([ 0.,  1.,  0.,  0.,  1.])}], [{'y/test/pred': array([ 1.,  0.,  0.,  1.,  0.]), 'y/train/pred': array([ 1.,  1.,  1.,  0.,  0.,  0.,  0.,  1.,  1.,  1.]), 'y/test/true': array([ 1.,  0.,  0.,  0.,  1.])}, {'y/test/pred': array([ 1.,  0.,  0.,  0.,  1.]), 'y/train/pred': array([ 1.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  1.,  1.]), 'y/test/true': array([ 1.,  0.,  0.,  0.,  1.])}], [{'y/test/pred': array([ 0.,  0.,  0.,  0.,  1.]), 'y/train/pred': array([ 0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  1.,  1.]), 'y/test/true': array([ 1.,  0.,  0.,  1.,  1.])}, {'y/test/pred': array([ 0.,  0.,  0.,  1.,  0.]), 'y/train/pred': array([ 0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  1.,  1.]), 'y/test/true': array([ 1.,  0.,  0.,  1.,  1.])}]]
    >>> cv.reduce()
    ResultSet(
    [{'key': SelectKBest/LDA/SVC, 'y/test/score_precision': [ 0.5         0.42857143], 'y/test/score_recall': [ 0.5         0.42857143], 'y/test/score_accuracy': 0.466666666667, 'y/test/score_f1': [ 0.5         0.42857143], 'y/test/score_recall_mean': 0.464285714286},
     {'key': SVC, 'y/test/score_precision': [ 0.7  0.8], 'y/test/score_recall': [ 0.875       0.57142857], 'y/test/score_accuracy': 0.733333333333, 'y/test/score_f1': [ 0.77777778  0.66666667], 'y/test/score_recall_mean': 0.723214285714}])
    >>> cv2 = CV(Methods(pipe2, SVC()), n_folds=3)
    >>> cv2.run(**Xy)
    [[{'y/test/pred': array([ 1.,  1.,  1.,  1.,  1.]), 'y/train/pred': array([ 1.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,  1.]), 'y/test/true': array([ 0.,  1.,  0.,  0.,  1.])}, {'y/test/pred': array([ 0.,  0.,  0.,  1.,  1.]), 'y/train/pred': array([ 1.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,  1.]), 'y/test/true': array([ 0.,  1.,  0.,  0.,  1.])}], [{'y/test/pred': array([ 1.,  1.,  1.,  1.,  1.]), 'y/train/pred': array([ 1.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  1.,  1.]), 'y/test/true': array([ 1.,  0.,  0.,  0.,  1.])}, {'y/test/pred': array([ 1.,  0.,  0.,  0.,  1.]), 'y/train/pred': array([ 1.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  1.,  1.]), 'y/test/true': array([ 1.,  0.,  0.,  0.,  1.])}], [{'y/test/pred': array([ 0.,  1.,  0.,  0.,  0.]), 'y/train/pred': array([ 0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  1.,  1.]), 'y/test/true': array([ 1.,  0.,  0.,  1.,  1.])}, {'y/test/pred': array([ 0.,  0.,  0.,  1.,  0.]), 'y/train/pred': array([ 0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  1.,  1.]), 'y/test/true': array([ 1.,  0.,  0.,  1.,  1.])}]]
    >>> cv2.reduce()
    ResultSet(
    [{'key': LDA/SVC, 'y/test/score_precision': [ 0.25        0.36363636], 'y/test/score_recall': [ 0.125       0.57142857], 'y/test/score_accuracy': 0.333333333333, 'y/test/score_f1': [ 0.16666667  0.44444444], 'y/test/score_recall_mean': 0.348214285714},
     {'key': SVC, 'y/test/score_precision': [ 0.7  0.8], 'y/test/score_recall': [ 0.875       0.57142857], 'y/test/score_accuracy': 0.733333333333, 'y/test/score_f1': [ 0.77777778  0.66666667], 'y/test/score_recall_mean': 0.723214285714}])

    """
    def __init__(self,
                 wrapped_node,
                 in_args_fit=None,
                 in_args_transform=None,
                 in_args_predict=None,
                 out_args_predict=None):
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
        is_fit_estimator = False
        if hasattr(wrapped_node, "fit") and hasattr(wrapped_node, "transform"):
            is_fit_estimator = True
        elif hasattr(wrapped_node, "fit") and hasattr(wrapped_node, "predict"):
            is_fit_estimator = True
        if not is_fit_estimator:
            raise ValueError("%s should implement fit and transform" %
                            wrapped_node.__class__.__name__)
        super(Estimator, self).__init__(wrapped_node=wrapped_node)
        if in_args_fit:
            self.in_args_fit = in_args_fit
        else:
            self.in_args_fit = _func_get_args_names(self.wrapped_node.fit)
        # Internal Estimator
        if hasattr(wrapped_node, "transform"):
            if in_args_transform:
                self.in_args_transform = in_args_transform
            else:
                self.in_args_transform = \
                    _func_get_args_names(self.wrapped_node.transform)
        # Leaf Estimator
        if hasattr(wrapped_node, "predict"):
            if in_args_predict:
                self.in_args_predict = in_args_predict
            else:
                self.in_args_predict = \
                    _func_get_args_names(self.wrapped_node.predict)
            if out_args_predict is None:
                fit_predict_diff = list(set(self.in_args_fit).difference(
                                            self.in_args_predict))
                if len(fit_predict_diff) > 0:
                    self.out_args_predict = fit_predict_diff
                else:
                    self.out_args_predict = self.in_args_predict
            else:
                self.out_args_predict = out_args_predict

    def _wrapped_node_transform(self, **Xy):
        Xy_out = _as_dict(self.wrapped_node.transform(
                            **_sub_dict(Xy, self.in_args_transform)),
                            keys=self.in_args_transform)
        return Xy_out

    def _wrapped_node_predict(self, **Xy):
        Xy_out = _as_dict(self.wrapped_node.predict(
                            **_sub_dict(Xy, self.in_args_predict)),
                            keys=self.out_args_predict)
        return Xy_out

    def transform(self, **Xy):
        """
        Parameter
        ---------
        Xy: dictionary
            parameters for fit and transform
        """
        is_fit_predict = False
        is_fit_transform = False
        if (hasattr(self.wrapped_node, "transform") and
                hasattr(self.wrapped_node, "predict")):
            if not self.children:
                # leaf node
                is_fit_predict = True
            else:
                # internal node
                is_fit_transform = True
        elif hasattr(self.wrapped_node, "transform"):
            is_fit_transform = True
        elif hasattr(self.wrapped_node, "predict"):
            is_fit_predict = True

        if is_fit_transform:
            if conf.KW_SPLIT_TRAIN_TEST in Xy:
                Xy_train, Xy_test = train_test_split(Xy)
                res = self.wrapped_node.fit(**_sub_dict(Xy_train,
                                                        self.in_args_fit))
                Xy_out_tr = self._wrapped_node_transform(**Xy_train)
                Xy_out_te = self._wrapped_node_transform(**Xy_test)
                Xy_out = train_test_merge(Xy_out_tr, Xy_out_te)
            else:
                res = self.wrapped_node.fit(**_sub_dict(Xy, self.in_args_fit))
                Xy_out = self._wrapped_node_transform(**Xy)
            # update ds with transformed values
            Xy.update(Xy_out)
            return Xy
        elif is_fit_predict:
            if conf.KW_SPLIT_TRAIN_TEST in Xy:
                Xy_train, Xy_test = train_test_split(Xy)
                Xy_out = dict()
                res = self.wrapped_node.fit(**_sub_dict(Xy_train,
                                            self.in_args_fit))
                Xy_out_tr = self._wrapped_node_predict(**Xy_train)
                Xy_out_tr = _dict_suffix_keys(Xy_out_tr,
                    suffix=conf.SEP + conf.TRAIN + conf.SEP + conf.PREDICTION)
                Xy_out.update(Xy_out_tr)
                # Test predict
                Xy_out_te = self._wrapped_node_predict(**Xy_test)
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
                Xy_out = self._wrapped_node_predict(**Xy)
                Xy_out = _dict_suffix_keys(Xy_out,
                    suffix=conf.SEP + conf.PREDICTION)
                ## True test
                Xy_true = _sub_dict(Xy, self.out_args_predict)
                Xy_out_true = _dict_suffix_keys(Xy_true,
                    suffix=conf.SEP + conf.TRUE)
                Xy_out.update(Xy_out_true)
            return Xy_out
        else:
            raise ValueError("%s should implement either transform or predict"
                            % self.wrapped_node.__class__.__name__)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
