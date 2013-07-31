"""
Spliters divide the work to do into several parallel sub-tasks.
They are of two types data spliters (CV, Perms) or tasks
splitter (Methods, Grid).


@author: edouard.duchesnay@cea.fr
@author: benoit.da_mota@inria.fr
"""

## Abreviations
## tr: train
## te: test
import collections
import numpy as np
import copy

from epac.workflow.base import BaseNode, key_push, key_pop
from epac.workflow.base import key_split
from epac.stores import StoreMem
from epac.utils import train_test_split
from epac.workflow.factory import NodeFactory
from epac.map_reduce.results import Result, ResultSet
from epac.utils import _list_indices, dict_diff, _sub_dict
from epac.map_reduce.reducers import ClassificationReport, PvalPerms
from epac.configuration import conf
from epac.workflow.wrappers import Wrapper

## ======================================================================== ##
## ==                                                                    == ##
## == Parallelization nodes
## ==
## ======================================================================== ##


# -------------------------------- #
# -- Splitter                   -- #
# -------------------------------- #

class BaseNodeSplitter(BaseNode):
    """Splitters are are non leaf node (degree >= 1) with children.
    They split the downstream data-flow to their children.
    They agregate upstream data-flow from their children.
    """
    def __init__(self):
        super(BaseNodeSplitter, self).__init__()

    def reduce(self, store_results=True):
        # Terminaison (leaf) node return results
        if not self.children:
            return self.load_results()
        # 1) Build sub-aggregates over children
        children_results = [child.reduce(store_results=False) for
            child in self.children]
        result_set = ResultSet(*children_results)
        if not self.reducer:
            return result_set
        # Group by key, without consideration of the fold/permutation number
        # which is the head of the key
        # use OrderedDict to preserve runing order
        from collections import OrderedDict
        groups = OrderedDict()
        for result in result_set:
            # remove the head of the key
            _, key_tail = key_pop(result["key"], index=0)
            result["key"] = key_tail
            if not key_tail in groups:
                groups[key_tail] = list()
            groups[key_tail].append(result)
        # For each key, stack results
        reduced = ResultSet()
        for key in groups:
            result_stacked = Result.stack(*groups[key])
            reduced.add(self.reducer.reduce(result_stacked))
        return reduced


class CV(BaseNodeSplitter):
    """Cross-validation parallelization.

    Parameters
    ----------
    node: Node | Estimator
        Estimator: should implement fit/predict/score function
        Node: Pipe | Par*

    n_folds: int
        Number of folds. (Default 5)

    cv_type: string
        Values: "stratified", "random", "loo". Default "stratified".

    random_state : int or RandomState
        Pseudo-random number generator state used for random sampling.

    reducer: Reducer
        A Reducer should inmplement the reduce(node, key2, val) method.
        Default ClassificationReport() with default arguments.
    """

    def __init__(self, node, n_folds=5, random_state=None,
                 cv_type="stratified", reducer=ClassificationReport(), **kwargs):
        super(CV, self).__init__()
        self.n_folds = n_folds
        self.random_state = random_state
        self.cv_type = cv_type
        self.reducer = reducer
        self.slicer = RowSlicer(signature_name="CV", nb=0, apply_on=None)
        self.children = VirtualList(size=n_folds, parent=self)
        self.slicer.parent = self
        subtree = NodeFactory.build(node)
        # subtree = node if isinstance(node, BaseNode) else LeafEstimator(node)
        self.slicer.add_child(subtree)

    def move_to_child(self, nb):
        self.slicer.set_nb(nb)
        if hasattr(self, "_sclices"):
            cpt = 0
            for train, test in self._sclices:
                if cpt == nb:
                    break
                cpt += 1
            self.slicer.set_sclices({conf.TRAIN: train, conf.TEST: test})
        return self.slicer

    def transform(self, **Xy):
        # Set the slicing
        if not "y" in Xy:
            raise ValueError('"y" should be provided')
        if self.cv_type == "stratified":
            if not self.n_folds:
                raise ValueError('"n_folds" should be set')
            from sklearn.cross_validation import StratifiedKFold
            self._sclices = StratifiedKFold(y=Xy["y"], n_folds=self.n_folds)
        elif self.cv_type == "random":
            if not self.n_folds:
                raise ValueError('"n_folds" should be set')
            from sklearn.cross_validation import KFold
            self._sclices = KFold(n=Xy["y"].shape[0], n_folds=self.n_folds,
                           random_state=self.random_state)
        elif self.cv_type == "loo":
            from sklearn.cross_validation import LeaveOneOut
            self._sclices = LeaveOneOut(n=Xy["y"].shape[0])
        return Xy

    def get_parameters(self):
        return dict(n_folds=self.n_folds)


class Perms(BaseNodeSplitter):
    """Permutation parallelization.

    Parameters
    ----------
    node: Node | Estimator
        Estimator: should implement fit/predict/score function
        Node: Pipe | Par*

    n_perms: int
        Number permutations.

    reducer: Reducer
        A Reducer should inmplement the reduce(key2, val) method.

    permute: string
        The name of the data to be permuted (default "y").

    random_state : int or RandomState
        Pseudo-random number generator state used for random sampling.

    reducer: Reducer
        A Reducer should inmplement the reduce(key2, val) method.
    """
    def __init__(self, node, n_perms=100, permute="y", random_state=None,
                 reducer=PvalPerms(), **kwargs):
        super(Perms, self).__init__()
        self.n_perms = n_perms
        self.permute = permute  # the name of the bloc to be permuted
        self.random_state = random_state
        self.reducer = reducer
        self.slicer = RowSlicer(signature_name="Perm", nb=0, apply_on=permute)
        self.children = VirtualList(size=n_perms, parent=self)
        self.slicer.parent = self
        subtree = NodeFactory.build(node)
        # subtree = node if isinstance(node, BaseNode) else LeafEstimator(node)
        self.slicer.add_child(subtree)

    def move_to_child(self, nb):
        self.slicer.set_nb(nb)
        if hasattr(self, "_sclices"):
            cpt = 0
            for perm in self._sclices:
                if cpt == nb:
                    break
                cpt += 1
            self.slicer.set_sclices(perm)
        return self.slicer

    def get_parameters(self):
        return dict(n_perms=self.n_perms, permute=self.permute)

    def transform(self, **Xy):
        # Set the slicing
        if not self.permute in Xy:
            raise ValueError('"%s" should be provided' % self.permute)
        from epac.sklearn_plugins import Permutations
        self._sclices = Permutations(n=Xy[self.permute].shape[0],n_perms=self.n_perms,
                                random_state=self.random_state)
        return Xy


class Methods(BaseNodeSplitter):
    """Parallelization is based on several runs of different methods
    """
    def __init__(self, *nodes):
        super(Methods, self).__init__()
        for node in nodes:
            node_cp = copy.deepcopy(node)
            node_cp = NodeFactory.build(node_cp)
            self.add_child(node_cp)
        curr_nodes = self.children
        leaves_key = [l.get_key() for l in self.walk_leaves()]
        curr_nodes_key = [c.get_key() for c in curr_nodes]
        while len(leaves_key) != len(set(leaves_key)) and curr_nodes:
            curr_nodes_state = [c.get_parameters() for c in curr_nodes]
            curr_nodes_next = list()
            for key in set(curr_nodes_key):
                collision_indices = _list_indices(curr_nodes_key, key)
                if len(collision_indices) == 1:  # no collision for this cls
                    continue
                diff_arg_keys = dict_diff(*[curr_nodes_state[i] for i
                                            in collision_indices]).keys()
                for curr_node_idx in collision_indices:
                    if diff_arg_keys:
                        curr_nodes[curr_node_idx].signature_args = \
                            _sub_dict(curr_nodes_state[curr_node_idx],
                                      diff_arg_keys)
                    curr_nodes_next += curr_nodes[curr_node_idx].children
            curr_nodes = curr_nodes_next
            curr_nodes_key = [c.get_key() for c in curr_nodes]
            leaves_key = [l.get_key() for l in self.walk_leaves()]
        leaves_key = [l.get_key() for l in self.walk_leaves()]
        if len(leaves_key) != len(set(leaves_key)):
            raise ValueError("Some methods are identical, they could not be "
                    "differentiated according to their arguments")

    def transform(self, **Xy):
        return Xy

    def reduce(self, store_results=True):
        # 1) Build sub-aggregates over children
        children_results = [child.reduce(store_results=False) for
            child in self.children]
        results = ResultSet(*children_results)
        if self.reducer:
            return self.reducer.reduce(results)
        return results

# -------------------------------- #
# -- Slicers                    -- #
# -------------------------------- #


class VirtualList(collections.Sequence):
    def __init__(self, size, parent):
        self.size = size
        self.parent = parent

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        if i >= self.size:
            raise IndexError("%s index out of range" % self.__class__.__name__)
        return self.parent.move_to_child(nb=i)
        #return self.parent.move_to_child(i, self.slicer)

    def __iter__(self):
        """ Iterate over leaves"""
        for i in xrange(self.size):
            yield self.__getitem__(i)

    def append(self, o):
        pass


class Slicer(BaseNode):
    """ Slicers are Splitters' children, they re-sclice the downstream blocs.
    """
    def __init__(self, signature_name, nb):
        super(Slicer, self).__init__()
        self.signature_name = signature_name
        self.signature_args = dict(nb=nb)

    def set_nb(self, nb):
        self.signature_args["nb"] = nb

    def get_parameters(self):
        return dict(slices=self.slices)

    def get_signature(self, nb=1):
        """Overload the base name method: use self.signature_name"""
        return self.signature_name + \
            "(nb=" + str(self.signature_args["nb"]) + ")"

    def get_signature_args(self):
        """overried get_signature_args to return a copy"""
        return copy.copy(self.signature_args)

    def reduce(self, store_results=True):
        results = ResultSet(self.children[0].reduce(store_results=False))
        for result in results:
            result["key"] = key_push(self.get_signature(), result["key"])
        return results


class RowSlicer(Slicer):
    """Row-wise reslicing of the downstream blocs.

    Parameters
    ----------
    name: string

    apply_on: string or list of strings (or None)
        The name(s) of the downstream blocs to be rescliced. If
        None, all downstream blocs are rescliced.
    """

    def __init__(self, signature_name, nb, apply_on):
        super(RowSlicer, self).__init__(signature_name, nb)
        self.slices = None
        self.n = 0  # the dimension of that array in ds should respect
        if not apply_on: # None is an acceptable value here
            self.apply_on = apply_on
        elif isinstance(apply_on, list):
            self.apply_on = apply_on
        elif isinstance(apply_on, str):
            self.apply_on = [apply_on]
        else:
            raise ValueError("apply_on must be a string or a list of strings or None")

    def set_sclices(self, slices):
        # convert as a list if required
        if isinstance(slices, dict):
            self.slices =\
                {k: slices[k].tolist() if isinstance(slices[k], np.ndarray)
                else slices[k] for k in slices}
            self.n = np.sum([len(v) for v in self.slices.values()])
        else:
            self.slices = \
                slices.tolist() if isinstance(slices, np.ndarray) else slices
            self.n = len(self.slices)

    def transform(self, **Xy):
        if not self.slices:
            raise ValueError("Slicing hasn't been initialized. "
            "Slicers constructors such as CV or Perm should be called "
            "with a sample. Ex.: CV(..., y=y), Perm(..., y=y)")
        data_keys = self.apply_on if self.apply_on else Xy.keys()
        # filter out non-array or array with wrong dimension
        for k in data_keys:
            if not hasattr(Xy[k], "shape") or \
                Xy[k].shape[0] != self.n:
                data_keys.remove(k)
        for data_key in data_keys:  # slice input data
            dat = Xy.pop(data_key)
            if isinstance(self.slices, dict):
                Xy[conf.KW_SPLIT_TRAIN_TEST] = True
                for sample_set in self.slices:
                    if len(dat.shape) == 2:
                        Xy[key_push(data_key, sample_set)] = dat[self.slices[sample_set], :]
                    else:
                        Xy[key_push(data_key, sample_set)] = dat[self.slices[sample_set]]
            else:
                if len(dat.shape) == 2:
                    Xy[data_key] = dat[self.slices, :]
                else:
                    Xy[data_key] = dat[self.slices]
        return Xy


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
    >>> from epac.workflow.splitters import CVBestSearchRefit
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
        scores_opt = np.max(scores) if self.arg_max else np.min(scores)
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
        return self.load_results()

if __name__ == "__main__":
    import doctest
    doctest.testmod()
