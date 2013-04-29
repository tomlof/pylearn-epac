"""
Epac : Embarrassingly Parallel Array Computing

@author: edouard.duchesnay@cea.fr
@author: benoit.da_mota@inria.fr
"""
print __doc__

## Abreviations
## tr: train
## te: test

import numpy as np
import copy

from epac.workflow.base import WFNode
from epac.workflow.estimators import WFNodeEstimator
from epac.utils import _list_indices, dict_diff, _sub_dict

## ======================================================================== ##
## ==                                                                    == ##
## == Parallelization nodes
## ==
## ======================================================================== ##


# -------------------------------- #
# -- Splitter                   -- #
# -------------------------------- #

class WFNodeSplitter(WFNode):
    """Splitters are are non leaf node (degree >= 1) with children.
    They split the downstream data-flow to their children.
    They agregate upstream data-flow from their children.
    """
    def __init__(self):
        super(WFNodeSplitter, self).__init__()


class ParCV(WFNodeSplitter):
    """Cross-validation parallelization.

    Parameters
    ----------
    node: Node | Estimator
        Estimator: should implement fit/predict/score function
        Node: Seq | Par*

    n_folds: int
        Number of folds.

    y: array
        if an array is provided do a StratifiedKFold.

    n: int
       Do a KFold CV, or a LeaveOneOut if n==n_folds

    random_state : int or RandomState
        Pseudo-random number generator state used for random sampling.

    reducer: Reducer
        A Reducer should inmplement the reduce(node, key2, val) method.
    """
    SUFFIX_TRAIN = "train"
    SUFFIX_TEST = "test"

    def __init__(self, node, n_folds, random_state=None, reducer=None,
                 **kwargs):
        super(ParCV, self).__init__()
        self.n_folds = n_folds
        self.random_state = random_state
        self.reducer = reducer
        self.add_children([WFNodeRowSlicer(signature_name="CV", nb=nb,
                               apply_on=None) for nb in xrange(n_folds)])
        for split in self.children:
            node = copy.deepcopy(node)
            node = node if isinstance(node, WFNode) else WFNodeEstimator(node)
            split.add_child(node)
        if "y" in kwargs or "n" in kwargs:
            self.finalize_init(**kwargs)

    def finalize_init(self, **Xy):
        cv = None
        if "y" in Xy:
            from sklearn.cross_validation import StratifiedKFold
            cv = StratifiedKFold(y=Xy["y"], n_folds=self.n_folds)
        elif "n" in Xy:
            n = Xy["n"]
            if n > self.n_folds:
                from sklearn.cross_validation import KFold
                cv = KFold(n=n, n_folds=self.n_folds,
                           random_state=self.random_state)
            elif n == self.n_folds:
                from sklearn.cross_validation import LeaveOneOut
                cv = LeaveOneOut(n=n)
        if cv:
            nb = 0
            for train, test in cv:
                self.children[nb].set_sclices({ParCV.SUFFIX_TRAIN: train,
                                     ParCV.SUFFIX_TEST: test})
                nb += 1
        # propagate down-way
        if self.children:
            if "n" in Xy:
                for child in self.children:
                    Xy["n"] = len(child.slices[ParCV.SUFFIX_TRAIN])
                    child.finalize_init(**Xy)
            else:
                #ICI if n in *Xy n should be redifined
                [child.finalize_init(**Xy) for child in self.children]

    def get_state(self):
        return dict(n_folds=self.n_folds)


class ParPerm(WFNodeSplitter):
    """Permutation parallelization.

    Parameters
    ----------
    node: Node | Estimator
        Estimator: should implement fit/predict/score function
        Node: Seq | Par*

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
    def __init__(self, node, n_perms, permute="y", random_state=None,
                 reducer=None, **kwargs):
        super(ParPerm, self).__init__()
        self.n_perms = n_perms
        self.permute = permute  # the name of the bloc to be permuted
        self.random_state = random_state
        self.reducer = reducer
        self.add_children([WFNodeRowSlicer(signature_name="Perm", nb=nb,
                              apply_on=permute) for nb in xrange(n_perms)])
        for perm in self.children:
            node = copy.deepcopy(node)
            node = node if isinstance(node, WFNode) else WFNodeEstimator(node)
            perm.add_child(node)
        if "y" in kwargs:
            self.finalize_init(**kwargs)

    def finalize_init(self, **Xy):
        if not "y" in Xy:
            raise KeyError("y is not provided to finalize the initialization")
        y = Xy["y"]
        from epac.sklearn_plugins import Permutation
        nb = 0
        for perm in Permutation(n=y.shape[0], n_perms=self.n_perms,
                                random_state=self.random_state):
            self.children[nb].set_sclices(perm)
            nb += 1
        # propagate down-way
        if self.children:
            [child.finalize_init(**Xy) for child in self.children]

    def get_state(self):
        return dict(n_perms=self.n_perms, permute=self.permute)


#class ParMethods(WFNodeSplitter):
#    """Parallelization is based on several runs of different methods
#    """
#    def __init__(self, *nodes):
#        super(ParMethods, self).__init__()
#        for node in nodes:
#            node = copy.deepcopy(node)
#            node = node if isinstance(node, WFNode) else WFNodeEstimator(node)
#            self.add_child(node)
#        # detect collisions in children signature
#        signatures = [c.get_signature() for c in self.children]
#        if len(signatures) != len(set(signatures)):  # collision
#            # in this case complete the signature finding differences
#            # in children states and put it in the args attribute
#            child_signatures = [c.get_signature() for c in self.children]
#            child_states = [c.get_state() for c in self.children]
#            # iterate over each level to solve collision
#            for signature in set(child_signatures):
#                collision_indices = _list_indices(child_signatures, signature)
#                if len(collision_indices) == 1:  # no collision for this cls
#                    continue
#                # Collision: add differences in states in the signature_args
#                diff_arg_keys = dict_diff(*[child_states[i] for i
#                                            in collision_indices]).keys()
#                for child_idx in collision_indices:
#                    self.children[child_idx].signature_args = \
#                        _sub_dict(child_states[child_idx], diff_arg_keys)


class ParMethods(WFNodeSplitter):
    """Parallelization is based on several runs of different methods
    """
    def __init__(self, *nodes):
        super(ParMethods, self).__init__()
        for node in nodes:
            node = copy.deepcopy(node)
            node = node if isinstance(node, WFNode) else WFNodeEstimator(node)
            self.add_child(node)
        children = self.children
        children_key = [c.get_key() for c in children]
        # while collision, recursively explore children to avoid collision
        # adding arguments to signature
        while len(children_key) != len(set(children_key)) and children:
            children_state = [c.get_state() for c in children]
            for key in set(children_key):
                collision_indices = _list_indices(children_key, key)
                if len(collision_indices) == 1:  # no collision for this cls
                    continue
                diff_arg_keys = dict_diff(*[children_state[i] for i
                                            in collision_indices]).keys()
                if diff_arg_keys:
                    for child_idx in collision_indices:
                        children[child_idx].signature_args = \
                            _sub_dict(children_state[child_idx], diff_arg_keys)
                children_next = list()
                for c in children:
                    children_next += c.children
                children = children_next
                children_key = [c.get_key() for c in children]
        leaves_key = [l.get_key() for l in self.get_leaves()]
        if len(leaves_key) != len(set(leaves_key)):
            raise ValueError("Some methods are identical, they could not be "
                    "differentiated according to their arguments")


class ParGrid(ParMethods):
    """Similar to ParMethods except the way that the upstream data-flow is
    processed.
    """
    def __init__(self, *nodes):
        super(ParGrid, self).__init__(*nodes)
        # Set signature2_args_str to"*" to create collision between secondary
        # keys see WFNodeRowSlicer.get_signature()
        for c in self.children:
            c.signature2_args_str = "*"


# -------------------------------- #
# -- Slicers                    -- #
# -------------------------------- #

class WFNodeSlicer(WFNode):
    """ Slicers are Splitters' children, they re-sclice the downstream blocs.
    """
    def __init__(self):
        super(WFNodeSlicer, self).__init__()


class WFNodeRowSlicer(WFNodeSlicer):
    """Row-wise reslicing of the downstream blocs.

    Parameters
    ----------
    name: string

    apply_on: string or list of strings
        The name(s) of the downstream blocs to be rescliced. If
        None, all downstream blocs are rescliced.
    """

    def __init__(self, signature_name, nb, apply_on):
        super(WFNodeRowSlicer, self).__init__()
        self.signature_name = signature_name
        self.signature_args = dict(nb=nb)
        self.slices = None
        self.n = 0  # the dimension of that array in ds should respect
        self.apply_on = apply_on

    def finalize_init(self, **Xy):
        Xy = self.transform(recursion=False, sample_set="train", **Xy)
        # propagate down-way
        if self.children:
            [child.finalize_init(**Xy) for child in self.children]

    def get_state(self):
        return dict(slices=self.slices)

    def get_signature(self, nb=1):
        """Overload the base name method.
        - use self.signature_name
        - Provoks intermediary keys collision which trig aggregation."""
        if nb is 1:
            args_str = ",".join([str(k) + "=" + str(self.signature_args[k])
                             for k in self.signature_args])
            args_str = "(" + args_str + ")"
            return self.signature_name + args_str
        else:
            return self.signature_name + "(*)"

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

    def transform(self, recursion=True, sample_set=None, **Xy):
        if not self.slices:
            raise ValueError("Slicing hasn't been initialized. "
            "Slicers constructors such as CV or Perm should be called "
            "with a sample. Ex.: CV(..., y=y), Perm(..., y=y)")
        if recursion:
            return self.top_down(func_name="transform", recursion=recursion,
                                 **Xy)
        data_keys = self.apply_on if self.apply_on else Xy.keys()
        # filter out non-array or array with wrong dimension
        for k in data_keys:
            if not hasattr(Xy[k], "shape") or \
                Xy[k].shape[0] != self.n:
                data_keys.remove(k)
        for data_key in data_keys:  # slice input data
            if not data_key in Xy:
                continue
            if isinstance(self.slices, dict):
                if not sample_set:
                    raise ValueError("sample_set should be provided. "
                    "self.slices is a dict with several slices, one should "
                    "indiquates which slices to use among %s" %
                    self.slices.keys())
                indices = self.slices[sample_set]
            else:
                indices = self.slices
            Xy[data_key] = Xy[data_key][indices]
        return Xy

    def fit(self, recursion=True, **Xy):
        """Call transform with sample_set="train" """
        if recursion:
            return self.top_down(func_name="fit", recursion=recursion, **Xy)
        return self.transform(recursion=False, sample_set="train", **Xy)

    def predict(self, recursion=True, **Xy):
        """Call transform  with sample_set="test" """
        if recursion:
            return self.top_down(func_name="predict", recursion=recursion,
                                 **Xy)
        return self.transform(recursion=False, sample_set="test", **Xy)
