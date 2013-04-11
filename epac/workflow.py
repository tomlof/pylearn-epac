"""
Epac : Embarrassingly Parallel Array Computing

@author: edouard.duchesnay@cea.fr
"""
print __doc__

## Abreviations
## tr: train
## te: test

import re
import numpy as np
import copy
from abc import abstractmethod
from stores import get_store
from utils import _list_union_inter_diff, _list_indices, _list_diff
from utils import _list_of_dicts_2_dict_of_lists
from utils import _sub_dict, dict_diff, _as_dict, _dict_prefix_keys
from utils import _func_get_args_names


## ================================= ##
## == Key manipulation utils      == ##
## ================================= ##

def key_split(key):
    """Split the key in in two parts: [protocol, path]

    Example
    -------
    >>> key_split('file:///tmp/toto')
    ['file', '/tmp/toto']
    """
    return key.split(conf.KEY_PROT_PATH_SEP, 1)


def key_join(prot="", path=""):
    """Join protocol and path to create a key

    Example
    -------
    >>> key_join("file", "/tmp/toto")
    'file:///tmp/toto'
    """
    return prot + conf.KEY_PROT_PATH_SEP + path


def key_pop(key):
    return key.rsplit(conf.KEY_PATH_SEP, 1)[0]


def key_push(key, basename):
    if key and basename:
        return key + conf.KEY_PATH_SEP + basename
    else:
        return key or basename


## ============================================== ##
## == down-stream data-flow manipulation utils == ##
## ============================================== ##

def xy_split(Xy):
    """Split Xy into two dictonaries. If input dictonnary whas not build
    with xy_merge(Xy1, Xy2) then return twice the input
    dictonnary.

    Parameters
    ----------
    Xy: dict

    Returns
    -------
    dict1, dict2 : splited dictionaries

    Example
    -------
    >>> xy_merged = xy_merge(dict(a=1, b=2), dict(a=33, b=44, c=55))
    >>> print xy_merged
    {'__2__%b': 44, '__2__%c': 55, '__2__%a': 33, '__1__%a': 1, '__1__%b': 2}
    >>> print xy_split(xy_merged)
    ({'a': 1, 'b': 2}, {'a': 33, 'c': 55, 'b': 44})
    >>> print xy_split(dict(a=1, b=2))
    ({'a': 1, 'b': 2}, {'a': 1, 'b': 2})
    """
    keys1 = [key1 for key1 in Xy if (str(key1).find("__1__%") == 0)]
    keys2 = [key2 for key2 in Xy if (str(key2).find("__2__%") == 0)]
    if not keys1 and not keys2:
        return Xy, Xy
    if keys1 and keys2:
        Xy1 = {key1.replace("__1__%", ""):
                        Xy[key1] for key1 in keys1}
        Xy2 = {key2.replace("__2__%", ""):
                        Xy[key2] for key2 in keys2}
        return Xy1, Xy2
    raise KeyError("data-flow could not be splitted")


def xy_merge(Xy1, Xy2):
    """Merge two dict avoiding keys collision.

    Parameters
    ----------
    Xy1: dict
    Xy2: dict

    Returns
    -------
    dict : merged dictionary

    Example
    -------
    >>> xy_merge(dict(a=1, b=2), dict(a=33, b=44, c=55))
    {'__2__%b': 44, '__2__%c': 55, '__2__%a': 33, '__1__%a': 1, '__1__%b': 2}
    """
    Xy1 = {"__1__%" + str(k): Xy1[k] for k in Xy1}
    Xy2 = {"__2__%" + str(k): Xy2[k] for k in Xy2}
    Xy1.update(Xy2)
    return Xy1


## ================================= ##
## == Configuration class         == ##
## ================================= ##

class conf:
    DEBUG = False
    VERBOSE = False
    STORE_FS_PICKLE_SUFFIX = ".pkl"
    STORE_FS_JSON_SUFFIX = ".json"
    STORE_NODE_PREFIX = "node"
    PREFIX_PRED = "pred_"
    PREFIX_TRUE = "true_"
    PREFIX_TEST = "test_"
    PREFIX_TRAIN = "train_"
    PREFIX_SCORE = "score_"
    KEY_PROT_MEM = "mem"  # key storage protocol: living object
    KEY_PROT_FS = "fs"  # key storage protocol: file system
    KEY_PATH_SEP = "/"
    KEY_PROT_PATH_SEP = "://"  # key storage protocol / path separator


class debug:
    current = None


## ======================================= ##
## == Workflow Node base abstract class == ##
## ======================================= ##

class WFNode(object):
    """WorkFlow Node base abstract class"""

    def __init__(self):
        self.parent = None
        self.children = list()
        self.store = ""
        # Results are indexed by intermediary keys. Each item is itself
        # a dictionnary
        self.results = dict()
        # The Key is the concatenation of nodes signatures from root to
        # Leaf.
        # Arguments are used to avoid collisions between keys.
        # In downstream flow collisions should always be avoided, so if
        # several children have the same name, use argument to sign the node.
        # In upstream flow collisions lead to aggregation of children node
        # with the same signature.
        self.signature_args = None  # dict of args to build the node signature
        self.combiner = None
        self.reducer = None

    def finalize_init(self, **Xy):
        """Overload this methods if init finalization is required"""
        if self.children:
            [child.finalize_init(**Xy) for child in self.children]

    # --------------------- #
    # -- Tree operations -- #
    # --------------------- #

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

    def add_children(self, children):
        for child in children:
            self.add_child(child)

    def get_leaves(self):
        if not self.children:
            return [self]
        else:
            leaves = []
            for child in self.children:
                leaves = leaves + child.get_leaves()
            return leaves

    def get_leftmost_leaf(self):
        """Return the left most leaf of a tree"""
        return self if not self.children else \
            self.children[0].get_leftmost_leaf()

    def get_rightmost_leaf(self):
        """Return the left most leaf of a tree"""
        return self if not self.children else \
            self.children[-1].get_rightmost_leaf()

    def get_node(self, key=None, regexp=None, stop_first_match=False):
        """Return a node given a key or a list of nodes given regexp

        Parameters
        ----------
        key: str

        regexp: str
            string with wild-cards: "*" to allow several matches.

        Example
        -------
        >>> from epac import ParCV, ParGrid, Seq
        >>> from sklearn.svm import SVC
        >>> from sklearn.lda import LDA
        >>> from sklearn.feature_selection import SelectKBest
        >>> y = [1, 1, 2, 2]
        >>> wf = ParCV(ParGrid(*[Seq(SelectKBest(k=k), SVC()) \
        ...     for k in [1, 5]]), n_folds=2, y=y)
        # List all leaves keys
        >>> for n in wf:
        ...     print n.get_key()
        ...
        ParCV/CV(nb=0)/ParGrid/SelectKBest(k=1)/SVC
        ParCV/CV(nb=0)/ParGrid/SelectKBest(k=5)/SVC
        ParCV/CV(nb=1)/ParGrid/SelectKBest(k=1)/SVC
        ParCV/CV(nb=1)/ParGrid/SelectKBest(k=5)/SVC
        # Get a single node using excat key match
        >>> wf.get_node(key="ParCV/CV(nb=0)/ParGrid/SelectKBest(k=1)").get_key()
        'ParCV/CV(nb=0)/ParGrid/SelectKBest(k=1)'
        # Get several nodes using wild cards
        >>> for n in wf.get_node(regexp="ParCV/*"):
        ...         print n.get_key()
        ...
        ParCV/CV(nb=0)
        ParCV/CV(nb=1)
        >>> for n in wf.get_node(regexp="*ParCV/CV(*)/*/*/SVC"):
        ...     print n.get_key()
        ...
        ParCV/CV(nb=0)/ParGrid/SelectKBest(k=1)/SVC
        ParCV/CV(nb=0)/ParGrid/SelectKBest(k=5)/SVC
        ParCV/CV(nb=1)/ParGrid/SelectKBest(k=1)/SVC
        ParCV/CV(nb=1)/ParGrid/SelectKBest(k=5)/SVC
        """
        if key:
            if key == self.get_key():
                return self
            for child in self.children:
                if key.find(child.get_key()) == 0:
                    return child.get_node(key)
        elif regexp:
            if isinstance(regexp, str):
                regexp = re.compile(regexp.replace("*", ".*").\
                             replace("(", "\(").replace(")", "\)"))
            if regexp.match(self.get_key()):
                return [self]
            nodes = list()
            for child in self.children:
                node = child.get_node(key=None, regexp=regexp,
                                      stop_first_match=stop_first_match)
                if node and stop_first_match:
                    return node[0] if isinstance(node, list) else node
                nodes += node
            return nodes
        else:
            raise ValueError("Provide at least a key for exact match"
            "or a regexp for wild card matches")

    def get_path_from_root(self):
        if self.parent is None:
            return [self]
        return self.parent.get_path_from_root() + [self]

    def get_path_from_node(self, node):
        if self is node:
            return [self]
        return self.parent.get_path_from_node(node) + [self]

    def __iter__(self):
        """ Iterate over leaves"""
        for leaf in self.get_leaves():
            yield leaf

    # --------------------- #
    # -- Key             -- #
    # --------------------- #

    def get_key(self, nb=1):
        """Return primary or intermediate key.

        Primary key is a unique identifier of a node in the tree, it is
        used to store leaf outputs at the end of the downstream flow.
        Intermediate key identify upstream results. Collisions between
        intermediate trig an aggregation of all results having the same key.

        Argument
        --------
        nb: int
            1 (default) return the primary key (ds data-flow).
            2 return the intermediate key (us data-flow).
        """
        if not self.parent:
            return key_push(self.store, self.get_signature(nb=nb))
        else:
            return key_push(self.parent.get_key(nb=nb),
                            self.get_signature(nb=nb))

    def get_signature(self, nb=1):
        """The signature of the current Node, used to build the key.

        By default primary and intermediate signatures are identical.
        Aggregation behavior of results during up-stream can be controled here:
        Redefine this method to create key collision and aggregation when nb=2.
        """
        if not self.signature_args:
            return self.__class__.__name__
        else:
            args_str = ",".join([str(k) + "=" + str(self.signature_args[k])
                             for k in self.signature_args])
            args_str = "(" + args_str + ")"
            return self.__class__.__name__ + args_str

    @abstractmethod
    def get_state(self):
        """Return the state of the object"""

    def add_results(self, key2=None, val_dict=None):
        """ Collect result output

        Parameters
        ----------
        key2 : (string) the intermediary key
        val_dict : dictionary of the intermediary value produced by the leaf
        nodes.
        """
        if not key2 in self.results:
            self.results[key2] = dict()
        self.results[key2].update(val_dict)

    # ------------------------------------------ #
    # -- Top-down data-flow operations        -- #
    # ------------------------------------------ #

    def top_down(self, func_name, recursion=True, **Xy):
        """Top-down data processing method

            This method does nothing more that recursively call
            parent/children func_name. Most of time, it should be re-defined.

            Parameters
            ----------
            func_name: str
                the name of the function to be called
            recursion: boolean
                if True recursively call parent/children func_name. If the
                current node is the root of the tree call the children.
                This way the whole tree is executed.
                If it is a leaf, then recursively call the parent before
                being executed. This a pipeline made of the path from the
                leaf to the root is executed.
            **Xy: dict
                the keyword dictionnary of data-flow

            Return
            ------
            A dictionnary of processed data
        """
        if conf.VERBOSE:
            print self.get_key(), func_name
        if conf.DEBUG:
            debug.current = self
            debug.Xy = Xy
        func = getattr(self, func_name)
        Xy = func(recursion=False, **Xy)
        #Xy = self.transform(**Xy)
        if recursion and self.children:
            # Call children func_name down to leaves
            ret = [child.top_down(func_name=func_name, recursion=recursion,
                            **Xy) for child in self.children]
            Xy = ret[0] if len(ret) == 1 else ret
        return Xy

    def fit(self, recursion=True, **Xy):
        if recursion:
            return self.top_down(func_name="fit", recursion=recursion, **Xy)
        return Xy

    def transform(self, recursion=True, **Xy):
        if recursion:
            return self.top_down(func_name="transform", recursion=recursion,
                                 **Xy)
        return Xy

    def predict(self, recursion=True, **Xy):
        if recursion:
            return self.top_down(func_name="predict", recursion=recursion,
                                 **Xy)
        return Xy

    def fit_predict(self, recursion=True, **Xy):
        if recursion:  # fit_predict was called in a top-down recursive context
            return self.top_down(func_name="fit_predict", recursion=recursion,
                                 **Xy)
        Xy_train, Xy_test = xy_split(Xy)
        Xy_train = self.fit(recursion=False, **Xy_train)
        Xy_test = self.predict(recursion=False, **Xy_test)
        if self.children:
            return xy_merge(Xy_train, Xy_test)
        else:
            return Xy_test

    # --------------------------------------------- #
    # -- Bottum-up data-flow operations (reduce) -- #
    # --------------------------------------------- #

    def bottum_up(self, store_results=True):
        if conf.DEBUG:
            debug.current = self
        # Terminaison (leaf) node return results
        if not self.children:
            if conf.DEBUG and conf.VERBOSE:
                print self.get_key(), self.results
            return self.results
        # 1) Build sub-aggregates over children
        children_results = [child.bottum_up(store_results=False) for
            child in self.children]
        if len(children_results) == 1:
            if store_results:
                self.add_results(self.get_key(2), children_results[0])
            return children_results[0]
        # 2) Test collision between intermediary keys
        keys_all = list()
        keys_set = set()
        for r in children_results:
            keys_all += r.keys()
            keys_set.update(r.keys())
        # 3) No collision: merge results in a lager dict an return
        if len(keys_set) == len(keys_all):
            merge = dict()
            [merge.update(item) for item in children_results]
            if store_results:
                [self.add_results(key2, merge[key2]) for key2 in merge]
            if conf.DEBUG and conf.VERBOSE:
                print self.get_key(), merge
            return merge
        # 4) Collision occurs
        # Aggregate (stack) all children results with identical
        # intermediary key, by stacking them according to
        # argumnents. Ex.: stack for a CV stack folds, for a ParGrid
        children_args = [child.signature_args for child in self.children]
        _, arg_names, diff_arg_names = _list_union_inter_diff(*[d.keys()
                                                for d in children_args])
        if diff_arg_names:
            raise ValueError("Children have different arguments name")
        sub_arg_names, sub_arg_values, results =\
            self._stack_results_over_argvalues(arg_names, children_results,
                                          children_args)
        # Reduce results if there is a reducer
        if self.reducer:
            results = {key2: self.reducer.reduce(self, key2, results[key2]) for
                key2 in results}
        if store_results:
            [self.add_results(key2, results[key2]) for key2 in results]
        if conf.DEBUG and conf.VERBOSE:
            print self.get_key(), results
        return results

    def _stack_results_over_argvalues(self, arg_names, children_results,
                                      children_args):
        if not arg_names:
            # Should correspond to a single results
            if len(children_results) != 1:
                raise ValueError("Many results were only one expected")
            return [], [], children_results[0]
        arg_name = arg_names[0]
        stack_arg = list()
        children_arg = [child_arg[arg_name] for child_arg in children_args]
        arg_values = list(set(sorted(children_arg)))
        for val in arg_values:
            children_select = _list_indices(children_arg, val)
            children_results_select = [children_results[i] for i
                                            in children_select]
            children_args_select = [children_args[i] for i in children_select]
            sub_arg_names, sub_arg_values, sub_stacked = \
                self._stack_results_over_argvalues(
                              arg_names=arg_names[1:],
                              children_results=children_results_select,
                              children_args=children_args_select)
            stack_arg.append(sub_stacked)
        stacked = _list_of_dicts_2_dict_of_lists(stack_arg,
                                                 axis_name=arg_name,
                                                 axis_values=arg_values)
        return arg_names, arg_values, stacked

    reduce = bottum_up

    # -------------------------------- #
    # -- I/O persistance operations -- #
    # -------------------------------- #

    def save(self, store=None, attr=None, recursion=True):
        """I/O (persistance) operation: save the node on the store. By default
        save the entire node. If attr is provided save only this attribute
        if non empty.

        Parameters
        ----------
        store: str
            This string allow to retrieve the store (see get_store(key)).

        attr: str
            Name of the Node's attribute to store, if provided only the
            attribute is saved, by default (None) the whole node is saved.

        recursion: bool
            Indicates if node should be recursively saved down to
            the leaves . Default (True).
        """
        if conf.DEBUG:
            #global _N
            debug.current = self
        if store:
            if len(key_split(store)) < 2:  # no store provided default use fs
                store = key_join(conf.KEY_PROT_FS, store)
            self.store = store
        if not self.store and not self.parent:
            raise ValueError("No store has been defined")
        key = self.get_key()
        store = get_store(key)
        if not attr:  # save the entire node
            # Prevent recursion saving of children/parent in a single dump:
            # replace reference to chidren/parent by basename strings
            clone = copy.copy(self)
            clone.children = [child.get_signature() for child in self.children]
            if self.parent:
                clone.parent = ".."
            if hasattr(self, "estimator"):  # Always pickle estimator
                clone.estimator = None
                store.save(self.estimator, key_push(key, "estimator"),
                           protocol="bin")
            store.save(clone, key=key_push(key, conf.STORE_NODE_PREFIX))
        else:
            o = self.__dict__[attr]
            # avoid saving attributes of len 0
            if not hasattr(o, "__len__") or (len(o) > 0):
                store.save(o, key_push(key, attr))
        if recursion and self.children:
            # Call children save down to leaves
            [child.save(attr=attr, recursion=recursion) for child
                in self.children]

    @classmethod
    def load(cls, key=None, store=None, recursion=True):
        """I/O (persistance) load a node indexed by key from the store.

        Parameters
        ----------
        key: string
            Load the node indexed by its key from the store. If missing then
            assume file system store and the key will point on the root of the
            store.

        store: string
            For fs store juste indicate the path to the store.

        recursion: int, boolean
            Indicates if node should be recursively loaded down to
            the leaves . Default (True).
        """
        if key is None:  # assume fs store, and point on the root of the store
            key = key_join(prot=conf.KEY_PROT_FS, path=store)
        store = get_store(key)
        loaded = store.load(key)
        node = loaded.pop(conf.STORE_NODE_PREFIX)
        node.__dict__.update(loaded)
        # Check for attributes to load
        if recursion and node.children:
            children = node.children
            node.children = list()
            for child in children:
                child_key = key_push(key, child)
                node.add_child(WF.load(key=child_key, recursion=recursion))
        return node

WF = WFNode


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

    def __repr__(self):
        return '%s(estimator=%s)' % (self.__class__.__name__,
            self.estimator.__repr__())

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
    task: Node | Estimator
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

    def __init__(self, task, n_folds, random_state=None, reducer=None,
                 **kwargs):
        super(ParCV, self).__init__()
        self.n_folds = n_folds
        self.random_state = random_state
        self.reducer = reducer
        self.add_children([WFNodeRowSlicer(signature_name="CV", nb=nb,
                               apply_on=None) for nb in xrange(n_folds)])
        for split in self.children:
            task = copy.deepcopy(task)
            task = task if isinstance(task, WFNode) else WFNodeEstimator(task)
            split.add_child(task)
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
    task: Node | Estimator
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
    def __init__(self, task, n_perms, permute="y", random_state=None,
                 reducer=None, **kwargs):
        super(ParPerm, self).__init__()
        self.n_perms = n_perms
        self.permute = permute  # the name of the bloc to be permuted
        self.random_state = random_state
        self.reducer = reducer
        self.add_children([WFNodeRowSlicer(signature_name="Perm", nb=nb,
                              apply_on=permute) for nb in xrange(n_perms)])
        for perm in self.children:
            task = copy.deepcopy(task)
            task = task if isinstance(task, WFNode) else WFNodeEstimator(task)
            perm.add_child(task)
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


class ParMethods(WFNodeSplitter):
    """Parallelization is based on several runs of different methods
    """
    def __init__(self, *args):
        super(ParMethods, self).__init__()
        for task in args:
            task = copy.deepcopy(task)
            task = task if isinstance(task, WFNode) else WFNodeEstimator(task)
            self.add_child(task)
        # detect collisions in children signature
        signatures = [c.get_signature() for c in self.children]
        if len(signatures) != len(set(signatures)):  # collision
            # in this case complete the signature finding differences
            # in children states and put it in the args attribute
            child_signatures = [c.get_signature() for c in self.children]
            child_states = [c.get_state() for c in self.children]
            # iterate over each level to solve collision
            for signature in set(child_signatures):
                collision_indices = _list_indices(child_signatures, signature)
                if len(collision_indices) == 1:  # no collision for this cls
                    continue
                # Collision: add differences in states in the signature_args
                diff_arg_keys = dict_diff(*[child_states[i] for i
                                            in collision_indices]).keys()
                for child_idx in collision_indices:
                    self.children[child_idx].signature_args = \
                        _sub_dict(child_states[child_idx], diff_arg_keys)


class ParGrid(ParMethods):
    """Similar to ParMethods except the way that the upstream data-flow is
    processed.
    """
    def __init__(self, *args):
        super(ParGrid, self).__init__(*args)
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


class CVGridSearchRefit(WFNodeEstimator):
    """CV grid search reducer

    Average results over first axis, then find the arguments that maximize or
    minimise a "score" over other axis.

    Parameters
    ----------
    key3: str
        a regular expression that match the score name to be oprimized.
        Default is "test.+%s"
    """ % conf.PREFIX_SCORE

    def __init__(self, task, n_folds, random_state=None, reducer=None,
                 key3="test.+" + conf.PREFIX_SCORE,
                 arg_max=True, **kwargs):
        super(CVGridSearchRefit, self).__init__(estimator=None)
        cv = ParCV(task=task, n_folds=n_folds, random_state=random_state,
                   reducer=reducer, **kwargs)
        self.key3 = key3
        self.arg_max = arg_max
        self.add_child(cv)  # first child is the CV

    def get_signature(self, nb=1):
        return self.__class__.__name__

    def fit(self, recursion=True, **Xy):
        # Fit/predict CV grid search
        cv_grid_search = self.children[0]
        cv_grid_search.fit_predict(recursion=True, **Xy)
        #  Pump-up results
        methods = list()
        cv_grid_search.bottum_up(store_results=True)
        for key2 in cv_grid_search.results:
            print key2
            pipeline = self.cv_grid_search(key2=key2,
                                           result=cv_grid_search.results[key2],
                                           cv_node=cv_grid_search)
            methods.append(pipeline)
        # Add children
        to_refit = ParMethods(*methods)
        if len(self.children) == 1:
            self.add_child(to_refit)
        else:
            self.children[1] = to_refit
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
            estimators.append(new_estimator_node)
        # Build the sequential pipeline
        pipeline = Seq(*estimators)
        return pipeline

## ======================================================================== ##
## ==                                                                    == ##
## == Sequential nodes
## ==
## ======================================================================== ##

def Seq(*args):
    """
    Sequential execution of Nodes.

    Parameters
    ----------
    task [, task]*
    """
    # SEQ(WFNode [, WFNode]*)
    #args = _group_args(*args)
    root = None
    for task in args:
        #task = copy.deepcopy(task)
        curr = task if isinstance(task, WFNode) else WFNodeEstimator(task)
        if not root:
            root = curr
        else:
            prev.add_child(curr)
        prev = curr
    return root
