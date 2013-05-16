"""
Spliters divide the work to do into several parallel sub-tasks.
They are of two types data spliters (ParCV, ParPerm) or tasks
splitter (ParMethods, ParGrid).


@author: edouard.duchesnay@cea.fr
@author: benoit.da_mota@inria.fr
"""

## Abreviations
## tr: train
## te: test
import collections
import numpy as np
import copy

from epac.workflow.base import BaseNode
from epac.workflow.estimators import Estimator
from epac.utils import _list_indices, dict_diff, _sub_dict

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


class ParCV(BaseNodeSplitter):
    """Cross-validation parallelization.

    Parameters
    ----------
    node: Node | Estimator
        Estimator: should implement fit/predict/score function
        Node: Seq | Par*

    n_folds: int
        Number of folds.

    cv_type: string
        Values: "stratified", "random", "loo". Default "stratified".

    random_state : int or RandomState
        Pseudo-random number generator state used for random sampling.

    reducer: Reducer
        A Reducer should inmplement the reduce(node, key2, val) method.
    """
    SUFFIX_TRAIN = "train"
    SUFFIX_TEST = "test"

    def __init__(self, node, n_folds=None, random_state=None, reducer=None,
                 cv_type="stratified", **kwargs):
        super(ParCV, self).__init__()
        self.n_folds = n_folds
        self.random_state = random_state
        self.cv_type = cv_type
        self.reducer = reducer
        self._slicer = RowSlicer(signature_name="CV", nb=0, apply_on=None)
        self._slicer.parent = self
        subtree = node if isinstance(node, BaseNode) else Estimator(node)
        self._slicer.add_child(subtree)
        self.children = SlicerVirtualList(size=n_folds, parent=self)

    def move_to_child(self, nb):
        print "move_to_child", nb
        self._slicer.set_nb(nb)
        if hasattr(self, "_sclices"):
            cpt = 0
            for train, test in self._sclices:
                print cpt, train, test
                if cpt == nb:
                    break
                cpt += 1
            self._slicer.set_sclices({ParCV.SUFFIX_TRAIN: train,
                                             ParCV.SUFFIX_TEST: test})
        return self._slicer

#        self.add_children([RowSlicer(signature_name="CV", nb=nb,
#                               apply_on=None) for nb in xrange(n_folds)])
#        for split in self.children:
#            node_cp = copy.deepcopy(node)
#            node_cp = node_cp if isinstance(node_cp, BaseNode) else Estimator(node_cp)
#            split.add_child(node_cp)

    def fit(self, recursion=True, **Xy):
        """Call transform with sample_set="train" """
        if recursion:
            return self.top_down(func_name="fit", recursion=recursion, **Xy)
        return self.transform(recursion=False, **Xy)

    def transform(self, recursion=True, **Xy):
        if recursion:
            return self.top_down(func_name="transform", recursion=recursion,
                                 **Xy)
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
        #if cv:
#        nb = 0
#        for train, test in self._sclices:
#            self.children[nb].set_sclices({ParCV.SUFFIX_TRAIN: train,
#                                 ParCV.SUFFIX_TEST: test})
#            nb += 1
        return Xy

    def get_state(self):
        return dict(n_folds=self.n_folds)



class ParPerm(BaseNodeSplitter):
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
        self.add_children([RowSlicer(signature_name="Perm", nb=nb,
                              apply_on=permute) for nb in xrange(n_perms)])
        for perm in self.children:
            node_cp = copy.deepcopy(node)
            node_cp = node_cp if isinstance(node_cp, BaseNode) else Estimator(node_cp)
            perm.add_child(node_cp)

    def get_state(self):
        return dict(n_perms=self.n_perms, permute=self.permute)

    def fit(self, recursion=True, **Xy):
        """Call transform with sample_set="train" """
        if recursion:
            return self.top_down(func_name="fit", recursion=recursion, **Xy)
        return self.transform(recursion=False, **Xy)

    def transform(self, recursion=True, **Xy):
        if recursion:
            return self.top_down(func_name="transform", recursion=recursion,
                                 **Xy)
        # Set the slicing
        if not "y" in Xy:
            raise ValueError('"y" should be provided')
        from epac.sklearn_plugins import Permutation
        perms = Permutation(n=Xy["y"].shape[0], n_perms=self.n_perms,
                                random_state=self.random_state)
        nb = 0
        for perm in perms:
            self.children[nb].set_sclices(perm)
            nb += 1
        return Xy


class ParMethods(BaseNodeSplitter):
    """Parallelization is based on several runs of different methods
    """
    def __init__(self, *nodes):
        super(ParMethods, self).__init__()
        for node in nodes:
            node_cp = copy.deepcopy(node)
            node_cp = node_cp if isinstance(node_cp, BaseNode) else Estimator(node_cp)
            self.add_child(node_cp)
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
        leaves_key = [l.get_key() for l in self.walk_leaves()]
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
        # keys see RowSlicer.get_signature()
        for c in self.children:
            c.signature2_args_str = "*"


# -------------------------------- #
# -- Slicers                    -- #
# -------------------------------- #

class SlicerVirtualList(collections.Sequence):
    def __init__(self, size, parent):
        self.size = size
        self.parent = parent

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        if i >= self.size:
            raise IndexError("%s index out of range" % self.__class__.__name__)
        return self.parent.move_to_child(i)

    def __iter__(self):
        """ Iterate over leaves"""
        for i in xrange(self.size):
            yield self.__getitem__(i)


class Slicer(BaseNode):
    """ Slicers are Splitters' children, they re-sclice the downstream blocs.
    """
    def __init__(self, signature_name, nb):
        super(Slicer, self).__init__()
        self.signature_name = signature_name
        self.signature_args = dict(nb=nb)

    def set_nb(self, nb):
        self.signature_args["nb"] = nb

    def get_state(self):
        return dict(slices=self.slices)

    def get_signature(self, nb=1):
        """Overload the base name method.
        - Use self.signature_name
        - Cause intermediary keys collision which trig aggregation."""
        if nb is 1:
            return self.signature_name + "(nb=" + str(self.signature_args["nb"]) + ")"
        else:
            return self.signature_name + "(*)"

    def get_signature_args(self):
        """overried get_signature_args to return a copy"""
        return copy.copy(self.signature_args)


class RowSlicer(Slicer):
    """Row-wise reslicing of the downstream blocs.

    Parameters
    ----------
    name: string

    apply_on: string or list of strings
        The name(s) of the downstream blocs to be rescliced. If
        None, all downstream blocs are rescliced.
    """

    def __init__(self, signature_name, nb, apply_on):
        super(RowSlicer, self).__init__(signature_name, nb)
        self.slices = None
        self.n = 0  # the dimension of that array in ds should respect
        self.apply_on = apply_on

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
