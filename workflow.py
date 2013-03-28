"""
Epac : Embarrassingly Parallel Array Computing
"""
print __doc__

## Abreviations
## ds: downstream
## us: upstream
## tr: train
## te: test

import numpy as np
import copy
from abc import abstractmethod
from stores import get_store
from utils import _list_union_inter_diff, _list_indices, _list_diff
from utils import _list_of_dicts_2_dict_of_lists
from utils import _sub_dict, _dict_diff, _as_dict, _dict_prefix_keys
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

def ds_split(ds_kwargs):
    """Split ds_kwargs into two dictonaries. If input dictonnary whas not build
    with ds_merge(ds_kwargs1, ds_kwargs2) then return twice the input
    dictonnary.

    Parameters
    ----------
    ds_kwargs: dict

    Returns
    -------
    dict1, dict2 : splited dictionaries

    Example
    -------
    >>> ds_merged = ds_merge(dict(a=1, b=2), dict(a=33, b=44, c=55))
    >>> print ds_merged
    {'__2__%b': 44, '__2__%c': 55, '__2__%a': 33, '__1__%a': 1, '__1__%b': 2}
    >>> print ds_split(ds_merged)
    ({'a': 1, 'b': 2}, {'a': 33, 'c': 55, 'b': 44})
    >>> print ds_split(dict(a=1, b=2))
    ({'a': 1, 'b': 2}, {'a': 1, 'b': 2})
    """
    keys1 = [key1 for key1 in ds_kwargs if (str(key1).find("__1__%") == 0)]
    keys2 = [key2 for key2 in ds_kwargs if (str(key2).find("__2__%") == 0)]
    if not keys1 and not keys2:
        return ds_kwargs, ds_kwargs
    if keys1 and keys2:
        ds_kwargs1 = {key1.replace("__1__%", ""):
                        ds_kwargs[key1] for key1 in keys1}
        ds_kwargs2 = {key2.replace("__2__%", ""):
                        ds_kwargs[key2] for key2 in keys2}
        return ds_kwargs1, ds_kwargs2
    raise KeyError("data-flow could not be splitted")


def ds_merge(ds_kwargs1, ds_kwargs2):
    """Merge two dict avoiding keys collision.

    Parameters
    ----------
    ds_kwargs1: dict
    ds_kwargs2: dict

    Returns
    -------
    dict : merged dictionary

    Example
    -------
    >>> ds_merge(dict(a=1, b=2), dict(a=33, b=44, c=55))
    {'__2__%b': 44, '__2__%c': 55, '__2__%a': 33, '__1__%a': 1, '__1__%b': 2}
    """
    ds_kwargs1 = {"__1__%" + str(k): ds_kwargs1[k] for k in ds_kwargs1}
    ds_kwargs2 = {"__2__%" + str(k): ds_kwargs2[k] for k in ds_kwargs2}
    ds_kwargs1.update(ds_kwargs2)
    return ds_kwargs1


## ================================= ##
## == Configuration class         == ##
## ================================= ##

class conf:
    DEBUG = False
    VERBOSE = True
    STORE_FS_PICKLE_SUFFIX = ".pkl"
    STORE_FS_JSON_SUFFIX = ".json"
    STOREWFNode_PREFIX = "node"
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


RECURSION_UP = 1
RECURSION_DOWN = 2


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
        # In upstream flow, sometime we wants to avoid collision
        # (no aggregation) so we set this flag to True. Sometime (ParGrid)
        # we want to creat collision and agregate children of the same name.
        self.sign_upstream_with_args = True
        self.combiner = None
        self.reducer = None

    def finalize_init(self, **ds_kwargs):
        """Overload this methods if init finalization is required"""
        if self.children:
            [child.finalize_init(**ds_kwargs) for child in self.children]

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

    def get_node(self, key):
        """Return a node given a key"""
        print self.get_key(), key == self.get_key()
        if key == self.get_key():
            return self
        for child in self.children:
            if key.find(child.get_key()) == 0:
                return child.get_node(key)

    def get_path_from_root(self):
        if self.parent is None:
            return [self]
        return self.parent.get_path_from_root() + [self]

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

        By default primary and intermediate signatures are identical. If we
        want to change this behavior in order to trig agregation this function
        should be overloaded as in WFNodeSlicer.

        """
        if nb is 1:  # primary signature, always sign with args if presents
            args_str = self.get_signature_args_str()
            args_str = "(" + args_str + ")" if args_str else ""
            return self.get_signature_name() + args_str
        elif nb is 2:  # intermediate signature, test if args should be used.
            if self.sign_upstream_with_args:
                args_str = self.get_signature_args_str()
                args_str = "(" + args_str + ")" if args_str else ""
                return self.get_signature_name() + args_str
            else:
                return self.get_signature_name()

    def get_signature_name(self):
        """The name of the current node, used to build the signature"""
        return self.__class__.__name__

    def get_signature_args(self):
        return self.signature_args

    def get_signature_args_str(self):
        """The arguments names/values of the current node, used to build
        the signature"""
        if not self.signature_args:
            return ""
        else:
            return ",".join([str(k) + "=" + str(self.signature_args[k]) for k
                                in self.signature_args])

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

    def top_down(self, func_name, recursion=True, **ds_kwargs):
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
            **ds_kwargs: dict
                the keyword dictionnary of data flow

            Return
            ------
            A dictionnary of processed data
        """
        if conf.VERBOSE:
            print self.get_key(), func_name
        if conf.DEBUG:
            debug.current = self
            debug.ds_kwargs = ds_kwargs
        recursion = self.check_recursion(recursion)
        if recursion is RECURSION_UP:
            # recursively call parent func_name up to root
            ds_kwargs = self.parent.top_down(func_name=func_name,
                                             recursion=recursion, **ds_kwargs)
        func = getattr(self, func_name)
        ds_kwargs = func(recursion=False, **ds_kwargs)
        #ds_kwargs = self.transform(**ds_kwargs)
        if recursion is RECURSION_DOWN:
            # Call children func_name down to leaves
            ret = [child.top_down(func_name=func_name, recursion=recursion,
                            **ds_kwargs) for child in self.children]
            ds_kwargs = ret[0] if len(ret) == 1 else ret
        return ds_kwargs

    def fit(self, recursion=True, **ds_kwargs):
        if recursion:
            return self.top_down(func_name="fit", recursion=recursion,
                                 **ds_kwargs)
        return ds_kwargs

    def transform(self, recursion=True, **ds_kwargs):
        if recursion:
            return self.top_down(func_name="transform", recursion=recursion,
                                 **ds_kwargs)
        return ds_kwargs

    def predict(self, recursion=True, **ds_kwargs):
        if recursion:
            return self.top_down(func_name="predict", recursion=recursion,
                                 **ds_kwargs)
        return ds_kwargs

    def fit_predict(self, recursion=True, **ds_kwargs):
        if recursion:  # fit_predict was called in a top-down recursive context
            return self.top_down(func_name="fit_predict", recursion=recursion,
                                 **ds_kwargs)
        ds_kwargs_train, ds_kwargs_test = ds_split(ds_kwargs)
        ds_kwargs_train = self.fit(recursion=False, **ds_kwargs_train)
        ds_kwargs_test = self.predict(recursion=False, **ds_kwargs_test)
        if self.children:
            return ds_merge(ds_kwargs_train, ds_kwargs_test)
        else:
            return ds_kwargs_test

    def check_recursion(self, recursion):
        """ Check the way a recursion call can go.

            Parameter
            ---------
            recursion: int, bool
            if bool guess the way the recursion can go
            else do nothing an return recursion."""
        if recursion and type(recursion) is bool:
            if not self.children:
                recursion = RECURSION_UP
            elif not self.parent:
                recursion = RECURSION_DOWN
            else:
                raise ValueError("recursion is True, but the node is " + \
            "neither a leaf nor the root tree, it then not possible to " + \
            "guess if recurssion should go up or down")
        if recursion is RECURSION_UP and not self.parent:
            return False
        if recursion is RECURSION_DOWN and not self.children:
            return False
        return recursion

    # --------------------------------------------- #
    # -- Bottum-up data-flow operations (reduce) -- #
    # --------------------------------------------- #

    def reduce(self, store_results=True):
        if conf.DEBUG:
            debug.current = self
        # Terminaison (leaf) node return results
        if not self.children:
            if conf.DEBUG and conf.VERBOSE:
                print self.get_key(), self.results
            return self.results
        # 1) Build sub-aggregates over children
        children_results = [child.reduce(store_results=False) for
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
        # stack results over sevral values of each arguments
        children_name, children_args = zip(*[(child.get_signature_name(),
                                               child.get_signature_args())
                                               for child in self.children])
        # Check that children have the same name, and same argument name
        # ie.: they differ only on argument values
        if len(set(children_name)) != 1:
            raise ValueError("Children have different names")
        _, arg_names, diff_arg_names = _list_union_inter_diff(*[d.keys()
                                                for d in children_args])
        if diff_arg_names:
            raise ValueError("Children have different arguments name")
        sub_arg_names, sub_arg_values, results =\
            self._stack_results_over_argvalues(arg_names, children_results,
                                          children_args)
        # Reduce results if there is a reducer
        if self.reducer:
            results = {key2: self.reducer.reduce(key2, results[key2]) for
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
            RECURSION_UP or RECURSION_DOWN, indicates si node should be
            recursively saved up to the root (RECURSION_UP) or down to
            the leaves (RECURSION_DOWN). Default (True) try to guess up
            (if leaf) or down (if root).
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
            key = key_push(key, conf.STOREWFNode_PREFIX)
            store.save(clone, key)
        else:
            o = self.__dict__[attr]
            # avoid saving attributes of len 0
            if not hasattr(o, "__len__") or (len(o) > 0):
                store.save(o, key_push(key, attr))
        recursion = self.check_recursion(recursion)
        if recursion is RECURSION_UP:
            # recursively call parent save up to root
            self.parent.save(attr=attr, recursion=recursion)
        if recursion is RECURSION_DOWN:
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
            if True, load recursively, trying to guess the way: if node is a
            leaf then recursively load the path to root int a bottum-up maner.
            If the node is a root recursively load the whole tree in a top-down
            manner.
            if int should be: RECURSION_UP or RECURSION_DOWN
        """
        if key is None:  # assume fs store, and point on the root of the store
            key = key_join(prot=conf.KEY_PROT_FS, path=store)
        store = get_store(key)
        loaded = store.load(key)
        node = loaded.pop(conf.STOREWFNode_PREFIX)
        node.__dict__.update(loaded)
        # Check for attributes to load
        #attribs = store.load(key_push(key, conf.STORE_ATTRIB_PREFIX))
        #if len(attribs) > 1:
        # children contain basename string: Save the string a recursively
        # walk/load children
        recursion = node.check_recursion(recursion)
        if recursion is RECURSION_UP:
            parent_key = key_pop(key)
            parent = WF.load(key=parent_key, recursion=recursion)
            parent.add_child(node)
        if recursion is RECURSION_DOWN:
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

    def get_signature_name(self):
        return self.estimator.__class__.__name__

    def get_state(self):
        return self.estimator.__dict__

    def fit(self, recursion=True, **ds_kwargs):
        # fit was called in a top-down recursive context
        if recursion:
            return self.top_down(func_name="fit", recursion=recursion,
                                 **ds_kwargs)
        # Regular fit
        Xy_dict = _sub_dict(ds_kwargs, self.args_fit)
        self.estimator.fit(**Xy_dict)
        if not self.children:  # if not children compute scores
            train_score = self.estimator.score(**Xy_dict)
            y_pred_names = _list_diff(self.args_fit, self.args_predict)
            y_train_score_dict = _as_dict(train_score, keys=y_pred_names)
            _dict_prefix_keys(conf.PREFIX_TRAIN +
                              conf.PREFIX_SCORE, y_train_score_dict)
            y_train_score_dict = {conf.PREFIX_TRAIN + conf.PREFIX_SCORE +
                str(k): y_train_score_dict[k] for k in y_train_score_dict}
            self.add_results(self.get_key(2), y_train_score_dict)
        if self.children:  # transform downstream data-flow (ds) for children
            return self.transform(recursion=False, **ds_kwargs)
        else:
            return self

    def transform(self, recursion=True, **ds_kwargs):
        # transform was called in a top-down recursive context
        if recursion:
            return self.top_down(func_name="transform", recursion=recursion,
                                 **ds_kwargs)
        # Regular transform:
        # catch args_transform in ds, transform, store output in a dict
        trn_dict = _as_dict(self.estimator.transform(**_sub_dict(ds_kwargs,
                                             self.args_transform)),
                       keys=self.args_transform)
        # update ds with transformed values
        ds_kwargs.update(trn_dict)
        return ds_kwargs

    def predict(self, recursion=True, **ds_kwargs):
        # fit was called in a top-down recursive context
        if recursion:
            return self.top_down(func_name="predict", recursion=recursion,
                                 **ds_kwargs)
        if self.children:  # if children call transform
            return self.transform(recursion=False, **ds_kwargs)
        # leaf node: do the prediction
        X_dict = _sub_dict(ds_kwargs, self.args_predict)
        y_pred_arr = self.estimator.predict(**X_dict)
        y_pred_names = _list_diff(self.args_fit, self.args_predict)
        y_pred_dict = _as_dict(y_pred_arr, keys=y_pred_names)
        results = _dict_prefix_keys(conf.PREFIX_PRED, y_pred_dict)
        # If true values are provided in ds then store them and compute scores
        if set(y_pred_names).issubset(set(ds_kwargs.keys())):
            y_true_dict = _sub_dict(ds_kwargs, y_pred_names)
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
        A Reducer should inmplement the reduce(key2, val) method.
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

    def finalize_init(self, **ds_kwargs):
        cv = None
        if "y" in ds_kwargs:
            from sklearn.cross_validation import StratifiedKFold
            cv = StratifiedKFold(y=ds_kwargs["y"], n_folds=self.n_folds)
        elif "n" in ds_kwargs:
            n = ds_kwargs["n"]
            if n > self.n_folds:
                from sklearn.cross_validation import KFold
                cv = KFold(n=n, n_folds=self.n_folds,
                           random_state=self.random_state)
            elif n == self.n_folds:
                from sklearn.cross_validation import LeaveOneOut
                cv = LeaveOneOut(n=n)
        if cv:
            print cv
            nb = 0
            for train, test in cv:
                self.children[nb].set_sclices({ParCV.SUFFIX_TRAIN: train,
                                     ParCV.SUFFIX_TEST: test})
                nb += 1
        # propagate down-way
        if self.children:
            if "n" in ds_kwargs:
                for child in self.children:
                    ds_kwargs["n"] = len(child.slices[ParCV.SUFFIX_TRAIN])
                    child.finalize_init(**ds_kwargs)
            else:
                #ICI if n in *ds_kwargs n should be redifined
                [child.finalize_init(**ds_kwargs) for child in
                    self.children]

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
        #print "y in kwargs", y in kwargs
        if "y" in kwargs:
            self.finalize_init(**kwargs)

    def finalize_init(self, **ds_kwargs):
        if not "y" in ds_kwargs:
            raise KeyError("y is not provided to finalize the initialization")
        y = ds_kwargs["y"]
        from epac.sklearn_plugins import Permutation
        nb = 0
        for perm in Permutation(n=y.shape[0], n_perms=self.n_perms, 
                                random_state=self.random_state):
            self.children[nb].set_sclices(perm)
            nb += 1
        # propagate down-way
        if self.children:
            [child.finalize_init(**ds_kwargs) for child in
                self.children]

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
            child_cls_str = [c.get_signature_name() for c in self.children]
            child_states = [c.get_state() for c in self.children]
            # iterate over each level to solve collision
            for cls in set(child_cls_str):
                collision_indices = _list_indices(child_cls_str, cls)
                if len(collision_indices) == 1:  # no collision for this cls
                    continue
                # Collision: add differences in states in the signature_args
                diff_arg_keys = _dict_diff(*[child_states[i] for i
                                            in collision_indices]).keys()
                for child_idx in collision_indices:
                    self.children[child_idx].signature_args = \
                        _sub_dict(child_states[child_idx], diff_arg_keys)


class ParGrid(ParMethods):
    """Similar to ParMethods except the way that the upstream data-flow is
    processed.
    """
    def __init__(self, *args):
        #methods = _group_args(*args)
        super(ParGrid, self).__init__(*args)
        for c in self.children:
            c.sign_upstream_with_args = False


# -------------------------------- #
# -- Slicers                    -- #
# -------------------------------- #

class WFNodeSlicer(WFNode):
    """ Slicers are Splitters' children, they re-sclice the downstream blocs.
    """
    def __init__(self):
        super(WFNodeSlicer, self).__init__()

    def get_signature(self, nb=1):
        """Overload for secondary key (us data-flow), return empty str."""
        if nb is 1:  # primary key (ds data-flow)
            return super(WFNodeSlicer, self).get_signature(nb=1)
        else:  # secondary key (us data-flow)
            return ""


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

    def finalize_init(self, **ds_kwargs):
        ds_kwargs = self.transform(recursion=False, sample_set="train",
                                   **ds_kwargs)
        # print self, "(",self.parent,")", self.slices, ds_kwargs
        # propagate down-way
        if self.children:
            [child.finalize_init(**ds_kwargs) for child in
                self.children]

    def get_state(self):
        return dict(slices=self.slices)

    def get_signature_name(self):
        """Overload the base name method"""
        return self.signature_name

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

    def transform(self, recursion=True, sample_set=None, **ds_kwargs):
        if not self.slices:
            raise ValueError("Slicing hasn't been initialized. "
            "Slicers constructors such as CV or Perm should be called "
            "with a sample. Ex.: CV(..., y=y), Perm(..., y=y)")
        if recursion:
            return self.top_down(func_name="transform", recursion=recursion,
                                 **ds_kwargs)
        data_keys = self.apply_on if self.apply_on else ds_kwargs.keys()
        # filter out non-array or array with wrong dimension
        for k in data_keys:
            if not hasattr(ds_kwargs[k], "shape") or \
                ds_kwargs[k].shape[0] != self.n:
                data_keys.remove(k)
        for data_key in data_keys:  # slice input data
            if not data_key in ds_kwargs:
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
            ds_kwargs[data_key] = ds_kwargs[data_key][indices]
        # print ds_kwargs
        return ds_kwargs

    def fit(self, recursion=True, **ds_kwargs):
        """Call transform with sample_set="train" """
        if recursion:
            return self.top_down(func_name="fit", recursion=recursion,
                                 **ds_kwargs)
        return self.transform(recursion=False, sample_set="train", **ds_kwargs)

    def predict(self, recursion=True, **ds_kwargs):
        """Call transform  with sample_set="test" """
        if recursion:
            return self.top_down(func_name="predict", recursion=recursion,
                                 **ds_kwargs)
        return self.transform(recursion=False, sample_set="test", **ds_kwargs)

## ======================================================================== ##
## ==                                                                    == ##
## == Sequential nodes
## ==
## ======================================================================== ##

def Seq(*args):
    """
    Sequential execution of tasks.

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
