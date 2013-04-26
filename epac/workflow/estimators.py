"""
Epac : Embarrassingly Parallel Array Computing

@author: edouard.duchesnay@cea.fr
@author: benoit.da_mota@inria.fr
"""
print __doc__

## Abreviations
## tr: train
## te: test

import re
import numpy as np
import copy
from epac.workflow.base import WFNode, conf, xy_split
from epac.utils import _as_dict, _dict_prefix_keys
from epac.utils import _func_get_args_names
from epac.utils import _sub_dict, _list_diff

## ================================= ##
## == Wrapper node for estimators == ##
## ================================= ##

class WFNodeEstimator(WFNode):
    """Node that wrap estimators"""

    def __init__(self, estimator):
        self.estimator = estimator
        self.signature2_args_str = None
        super(WFNodeEstimator, self).__init__()
        self.args_fit = _func_get_args_names(self.estimator.fit) \
            if hasattr(self.estimator, "fit") else None
        self.args_predict = _func_get_args_names(self.estimator.predict) \
            if hasattr(self.estimator, "predict") else None
        self.args_transform = _func_get_args_names(self.estimator.transform) \
            if hasattr(self.estimator, "transform") else None

    def get_signature(self, nb=1):
        """Overload the base name method.
        - Use estimator.__class__.__name__
        - If signature2_args_str is not None, use it. This way we have
        intermediary keys collision which trig aggregation."""
        if not self.signature_args:
            return self.estimator.__class__.__name__
        elif nb is 2 and self.signature2_args_str:
            return self.estimator.__class__.__name__ + "(" + \
                    self.signature2_args_str + ")"
        else:
            args_str = ",".join([str(k) + "=" + str(self.signature_args[k])
                             for k in self.signature_args])
            args_str = "(" + args_str + ")"
            return self.estimator.__class__.__name__ + args_str

    def get_state(self):
        return self.estimator.__dict__

    def fit(self, recursion=True, **Xy):
        # fit was called in a top-down recursive context
        if recursion:
            return self.top_down(func_name="fit", recursion=recursion, **Xy)
        # Regular fit
        Xy_dict = _sub_dict(Xy, self.args_fit)
        self.estimator.fit(**Xy_dict)
        if not self.children:  # if not children compute scores
            train_score = self.estimator.score(**Xy_dict)
            y_pred_names = _list_diff(self.args_fit, self.args_predict)
            y_train_score_dict = _as_dict(train_score, keys=y_pred_names)
            _dict_prefix_keys(conf.PREFIX_TRAIN + conf.PREFIX_SCORE,
                              y_train_score_dict)
            y_train_score_dict = {conf.PREFIX_TRAIN + conf.PREFIX_SCORE +
                str(k): y_train_score_dict[k] for k in y_train_score_dict}
            self.add_results(self.get_key(2), y_train_score_dict)
        if self.children:  # transform downstream data-flow (ds) for children
            return self.transform(recursion=False, **Xy)
        else:
            return self

    def transform(self, recursion=True, **Xy):
        # transform was called in a top-down recursive context
        if recursion:
            return self.top_down(func_name="transform", recursion=recursion,
                                 **Xy)
        # Regular transform:
        # catch args_transform in ds, transform, store output in a dict
        trn_dict = _as_dict(self.estimator.transform(**_sub_dict(Xy,
                                             self.args_transform)),
                       keys=self.args_transform)
        # update ds with transformed values
        Xy.update(trn_dict)
        return Xy

    def predict(self, recursion=True, **Xy):
        # fit was called in a top-down recursive context
        if recursion:
            return self.top_down(func_name="predict", recursion=recursion,
                                 **Xy)
        if self.children:  # if children call transform
            return self.transform(recursion=False, **Xy)
        # leaf node: do the prediction
        X_dict = _sub_dict(Xy, self.args_predict)
        y_pred_arr = self.estimator.predict(**X_dict)
        y_pred_names = _list_diff(self.args_fit, self.args_predict)
        y_pred_dict = _as_dict(y_pred_arr, keys=y_pred_names)
        results = _dict_prefix_keys(conf.PREFIX_PRED, y_pred_dict)
        # If true values are provided in ds then store them and compute scores
        if set(y_pred_names).issubset(set(Xy.keys())):
            y_true_dict = _sub_dict(Xy, y_pred_names)
            # compute scores
            X_dict.update(y_true_dict)
            test_score = self.estimator.score(**X_dict)
            y_test_score_dict = _as_dict(test_score, keys=y_pred_names)
            # prefix results keys by test_score_
            y_true_dict = _dict_prefix_keys(conf.PREFIX_TRUE, y_true_dict)
            results.update(y_true_dict)
            y_test_score_dict = _dict_prefix_keys(
                conf.PREFIX_TEST + conf.PREFIX_SCORE, y_test_score_dict)
            results.update(y_test_score_dict)
            self.add_results(self.get_key(2), results)
        return y_pred_arr




class ParCVGridSearchRefit(WFNodeEstimator):
    """Cross-validation + grid-search then refit with optimals parameters.

    Average results over first axis, then find the arguments that maximize or
    minimise a "score" over other axis.

    Parameters
    ----------

    See ParCV parameters

    key3: str
        a regular expression that match the score name to be optimized.
        Default is "test.+%s"

    arg_max: boolean
        True/False take parameters that maximize/minimize the score. Default
        is True.
    """ % conf.PREFIX_SCORE

    def __init__(self, *tasks, **kwargs):
        super(ParCVGridSearchRefit, self).__init__(estimator=None)
        key3 = kwargs.pop("key3") if "key3" in kwargs \
            else "test.+" + conf.PREFIX_SCORE
        arg_max = kwargs.pop("arg_max") if "arg_max" in kwargs else True
        from epac.workflow.splitters import ParCV, ParGrid
        grid = ParGrid(*tasks)
        cv = ParCV(task=grid, **kwargs)
        self.key3 = key3
        self.arg_max = arg_max
        self.add_child(cv)  # first child is the CV

    def get_signature(self, nb=1):
        return self.__class__.__name__

    def get_children_top_down(self):
        """Return children during the top-down exection."""
        return []

    def get_children_bottum_up(self):
        """Return children during the bottum-up execution."""
        return self.children[1]

    def fit(self, recursion=True, **Xy):
        # Fit/predict CV grid search
        cv_grid_search = self.children[0]
        cv_grid_search.fit_predict(recursion=True, **Xy)
        #  Pump-up results
        methods = list()
        cv_grid_search.bottum_up(store_results=True)
        for key2 in cv_grid_search.results:
            pipeline = self.cv_grid_search(key2=key2,
                                           result=cv_grid_search.results[key2],
                                           cv_node=cv_grid_search)
            methods.append(pipeline)
        # Add children
        from epac.workflow.splitters import ParMethods
        to_refit = ParMethods(*methods)
        if len(self.children) > 1:  # remove potential previously pipe-line
            self.children = self.children[:1]
        self.add_child(to_refit)
        to_refit.fit(recursion=True, **Xy)
        return self

    def predict(self, recursion=True, **Xy):
        """Call transform  with sample_set="test" """
        refited = self.children[1]
        return refited.predict(recursion=True, **Xy)

    def fit_predict(self, recursion=True, **Xy):
        Xy_train, Xy_test = xy_split(Xy)
        self.fit(recursion=False, **Xy_train)
        Xy_test = self.predict(recursion=False, **Xy_test)
        return Xy_test

    def cv_grid_search(self, key2, result, cv_node):
        #print node, key2, result
        match_key3 = [key3 for key3 in result
            if re.search(self.key3, str(key3))]
        if len(match_key3) != 1:
            raise ValueError("None or more than one tertiary key found")
        # 1) Retrieve pairs of optimal (argument-name, value)
        key3 = match_key3[0]
        grid_cv = result[key3]
        mean_cv = np.mean(np.array(grid_cv), axis=0)
        mean_cv_opt = np.max(mean_cv) if self.arg_max else  np.min(mean_cv)
        idx_best = np.where(mean_cv == mean_cv_opt)
        idx_best = [idx[0] for idx in idx_best]
        grid = grid_cv[0]
        args_best = list()
        while len(idx_best):
            idx = idx_best[0]
            args_best.append((grid.axis_name, grid.axis_values[idx]))
            idx_best = idx_best[1:]
            grid = grid[0]
        # Retrieve one node that match intermediary key
        leaf = cv_node.get_node(regexp=key2, stop_first_match=True)
        # get path from current node
        path = leaf.get_path_from_node(cv_node)
        estimators = list()
        for node in path:
            if not hasattr(node, "estimator"):  # Strip off non estimator nodes
                continue
            estimator_args = copy.deepcopy(node.signature_args)
            for i in xrange(len(estimator_args)):
                arg_best = args_best[0]
                args_best = args_best[1:]
                #print node.get_key(), "Set", arg_best[0], "=", arg_best[1]
                estimator_args[arg_best[0]] = arg_best[1]
            new_estimator = copy.deepcopy(node.estimator)
            new_estimator.__dict__.update(estimator_args)
            #print "=>", new_estimator
            new_estimator_node = WFNodeEstimator(new_estimator)
            new_estimator_node.signature_args = estimator_args
            # Parameters should no appear in intermediary key
            new_estimator_node.signature2_args_str = "*"
            estimators.append(new_estimator_node)
        # Build the sequential pipeline
        from epac.workflow.pipeline import Seq
        pipeline = Seq(*estimators)
        return pipeline
