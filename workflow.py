"""
Epac : Embarrassingly Parallel Array Computing
"""
print __doc__

## Abreviations
## ds: downstream
## us: upstream
## tr: train
## te: test


_VERBOSE = True
_DEBUG = True
_N = None  # global reference to the current Node usefull to debug

import numpy as np
import copy
from abc import abstractmethod
from stores import get_store
from utils import _list_union_inter_diff, _list_indices, _list_diff
from utils import _list_of_dicts_2_dict_of_lists
from utils import _sub_dict, _dict_diff, _as_dict, _dict_prefix_keys
from utils import _func_get_args_names


def key_split(key):
    """Split the key in in two parts: [protocol, path]

    Example
    -------
    >>> key_split('file:///tmp/toto')
    ['file', '/tmp/toto']
    """
    return key.split(Config.KEY_PROT_PATH_SEP, 1)


def key_join(prot="", path=""):
    """Join protocol and path to create a key

    Example
    -------
    >>> key_join("file", "/tmp/toto")
    'file:///tmp/toto'
    """
    return prot + Config.KEY_PROT_PATH_SEP + path


def key_pop(key):
    return key.rsplit(Config.KEY_PATH_SEP, 1)[0]


def key_push(key, basename):
    if key and basename:
        return key + Config.KEY_PATH_SEP + basename
    else:
        return key or basename


class Config:
    STORE_FS_PICKLE_SUFFIX = ".pkl"
    STORE_FS_JSON_SUFFIX = ".json"
    #STORE_RESULTS_PREFIX = "__result"
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


RECURSION_UP = 1
RECURSION_DOWN = 2


class _Node(object):
    """Nodes base class"""

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
        return self if not self.children else self.children[0].get_leftmost_leaf()

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
        should be overloaded as in _NodeSlicer.

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
        if _VERBOSE:
            print self.get_key(), func_name
        if _DEBUG:
            global _N
            _N = self
            self.ds_kwargs = ds_kwargs # self = leaf; ds_kwargs = self.ds_kwargs
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
        self.fit(recursion=False, **ds_kwargs)
        return self.predict(recursion=False, **ds_kwargs)

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
        if _DEBUG:
            global _N
            _N = self
        # Terminaison (leaf) node return results
        if not self.children:
            return self.results
        # 1) Build sub-aggregates over children
        children_results = [child.reduce(store_results=False) for
            child in self.children]
        if len(children_results) == 1:
            if store_results:
                self.add_results(self.get_key(2), children_results[0])
            return children_results[0]
        # 2) Test if for collision between intermediary keys
        keys_all = [r.keys() for r in children_results]
        np.sum([len(ks) for ks in keys_all])
        keys_set = set()
        [keys_set.update(ks) for ks in keys_all]
        # 3) If no collision , simply merge results in a lager dict an return
        # it
        if len(keys_set) == len(keys_all):
            merge = dict()
            [merge.update(item) for item in children_results]
            if store_results:
                [self.add_results(key2, merge[key2]) for key2 in merge]
            return merge
        # 4) Collision occurs
        # Aggregate (stack) all children results with identical
        # intermediary key, by stacking them according to
        # argumnents. Ex.: stack for a CV stack folds, for a ParGrid
        # stack results over sevral values of each arguments
        children_name, children_args = zip(*[(child.get_signature_name(),
                                               child.get_signature_args())
                                               for child in self.children])
        # Cheack that children have the same name, and same argument name
        # ie.: they differ only on argument values
        if len(set(children_name)) != 1:
            raise ValueError("Children of a Reducer have different names")
        _, arg_names, diff_arg_names = _list_union_inter_diff(*[d.keys()
                                                for d in children_args])
        if diff_arg_names:
            raise ValueError("Children of a Reducer have different arguements"
            "keys")
        sub_arg_names, sub_arg_values, results =\
            self._stack_results_over_argvalues(arg_names, children_results,
                                          children_args)
        # Reduce results if there is a reducer
        if self.reducer:
            results = {key2: self.reducer.reduce(key2, results[key2]) for
                key2 in results}
        if store_results:
            [self.add_results(key2, results[key2]) for key2 in results]
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
        if _DEBUG:
            global _N
            _N = self
        if store:
            if len(key_split(store)) < 2:  # no store provided default use fs
                store = key_join(Config.KEY_PROT_FS, store)
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
            key = key_push(key, Config.STORE_NODE_PREFIX)
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


def load_workflow(key=None, store=None, recursion=True):
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
        key = key_join(prot=Config.KEY_PROT_FS, path=store)
    store = get_store(key)
    loaded = store.load(key)
    node = loaded.pop(Config.STORE_NODE_PREFIX)
    node.__dict__.update(loaded)
    # Check for attributes to load
    #attribs = store.load(key_push(key, Config.STORE_ATTRIB_PREFIX))
    #if len(attribs) > 1:
    # children contain basename string: Save the string a recursively
    # walk/load children
    recursion = node.check_recursion(recursion)
    if recursion is RECURSION_UP:
        parent_key = key_pop(key)
        parent = load_workflow(key=parent_key, recursion=recursion)
        parent.add_child(node)
    if recursion is RECURSION_DOWN:
        children = node.children
        node.children = list()
        for child in children:
            child_key = key_push(key, child)
            node.add_child(load_workflow(key=child_key, recursion=recursion))
    return node


## ================================= ##
## == Wrapper node for estimators == ##
## ================================= ##

class _NodeEstimator(_Node):
    """Node that wrap estimators"""

    def __init__(self, estimator):
        self.estimator = estimator
        super(_NodeEstimator, self).__init__()
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
            _dict_prefix_keys(Config.PREFIX_TRAIN +
                              Config.PREFIX_SCORE, y_train_score_dict)
            y_train_score_dict = {Config.PREFIX_TRAIN + Config.PREFIX_SCORE +
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
        results = _dict_prefix_keys(Config.PREFIX_PRED, y_pred_dict)
        # If true values are provided in ds then store them and compute scores
        if set(y_pred_names).issubset(set(ds_kwargs.keys())):
            y_true_dict = _sub_dict(ds_kwargs, y_pred_names)
            # compute scores
            X_dict.update(y_true_dict)
            test_score = self.estimator.score(**X_dict)
            y_test_score_dict = _as_dict(test_score, keys=y_pred_names)
            # prefix results keys by test_score_
            y_true_dict = _dict_prefix_keys(Config.PREFIX_TRUE, y_true_dict)
            results.update(y_true_dict)
            y_test_score_dict = _dict_prefix_keys(
                Config.PREFIX_TEST + Config.PREFIX_SCORE, y_test_score_dict)
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

class _NodeSplitter(_Node):
    """Splitters are are non leaf node (degree >= 1) with children.
    They split the downstream data-flow to their children.
    They agregate upstream data-flow from their children.
    """
    def __init__(self):
        super(_NodeSplitter, self).__init__()


class ParCV(_NodeSplitter):
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
        self.add_children([_NodeRowSlicer(signature_name="CV", nb=nb,
                               apply_on=None) for nb in xrange(n_folds)])
        for split in self.children:
            task = copy.deepcopy(task)
            task = task if isinstance(task, _Node) else _NodeEstimator(task)
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


class ParPerm(_NodeSplitter):
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
        self.add_children([_NodeRowSlicer(signature_name="Perm", nb=nb,
                              apply_on=permute) for nb in xrange(n_perms)])
        for perm in self.children:
            task = copy.deepcopy(task)
            task = task if isinstance(task, _Node) else _NodeEstimator(task)
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
        for perm in Permutation(n=y.shape[0], n_perms=self.n_perms):
            self.children[nb].set_sclices(perm)
            nb += 1
        # propagate down-way
        if self.children:
            [child.finalize_init(**ds_kwargs) for child in
                self.children]

    def get_state(self):
        return dict(n_perms=self.n_perms, permute=self.permute)


class ParMethods(_NodeSplitter):
    """Parallelization is based on several runs of different methods
    """
    def __init__(self, *args):
        super(ParMethods, self).__init__()
        for task in args:
            task = copy.deepcopy(task)
            task = task if isinstance(task, _Node) else _NodeEstimator(task)
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

class _NodeSlicer(_Node):
    """ Slicers are Splitters' children, they re-sclice the downstream blocs.
    """
    def __init__(self):
        super(_NodeSlicer, self).__init__()

    def get_signature(self, nb=1):
        """Overload for secondary key (us data-flow), return empty str."""
        if nb is 1:  # primary key (ds data-flow)
            return super(_NodeSlicer, self).get_signature(nb=1)
        else:  # secondary key (us data-flow)
            return ""


class _NodeRowSlicer(_NodeSlicer):
    """Row-wise reslicing of the downstream blocs.

    Parameters
    ----------
    name: string

    apply_on: string or list of strings
        The name(s) of the downstream blocs to be rescliced. If
        None, all downstream blocs are rescliced.
    """

    def __init__(self, signature_name, nb, apply_on):
        super(_NodeRowSlicer, self).__init__()
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
    # SEQ(_Node [, _Node]*)
    #args = _group_args(*args)
    root = None
    for task in args:
        #task = copy.deepcopy(task)
        curr = task if isinstance(task, _Node) else _NodeEstimator(task)
        if not root:
            root = curr
        else:
            prev.add_child(curr)
        prev = curr
    return root
