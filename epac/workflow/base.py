"""
Base Workflow node plus keys manipulation utilities.

@author: edouard.duchesnay@cea.fr
@author: benoit.da_mota@inria.fr
"""


import re, sys
import copy
import numpy as np
from abc import abstractmethod
from epac.stores import get_store
from epac.utils import _list_union_inter_diff, _list_indices
from epac.utils import _list_of_dicts_2_dict_of_lists


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
    TRACE_TOPDOWN = False
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
    SUFFIX_JOB = "job"


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
        self.reducer = None

    def __repr__(self):
        return self.get_key()

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

    def get_all_nodes(self):
        if not self.children:
            return [self]
        else:
            nodes = [self]
            for child in self.children:
                nodes = nodes + child.get_all_nodes()
            return nodes

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

#    def __iter__(self):
#        """ Iterate over leaves"""
#        for node in self.get_all_nodes():
#            yield node
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
        elif not hasattr(self.parent, "get_key"):
            return key_push(str(self.parent), self.get_signature(nb=nb))
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

    def stats(self, group_by="key", sort_by="count"):
        """Statistics on the workflow
            Parameters
            ----------
            group_by: str
                Group by "key" or class name: "class" (default "key")
            sort_by: str
                Sort by "count" or "size" (default "count")
        """
        stat_dict = dict()
        nodes = self.get_all_nodes()
        for n in nodes:
            if group_by == "class":
                name = n.__class__.__name__
            else:
                name = n.get_signature()
            if not (name in stat_dict):
                stat_dict[name] = [0, 0]
            stat_dict[name][0] += 1
            stat_dict[name][1] += sys.getsizeof(n)
        stat = [(name, stat_dict[name][0], stat_dict[name][1]) for name in
                stat_dict]
        idx = 2 if sort_by == "size" else 1
        order = np.argsort([s[idx] for s in stat])[::-1]
        return [stat[i] for i in order]

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
        if conf.TRACE_TOPDOWN:
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
                            **Xy) for child in self.get_children_top_down()]
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

    def get_children_top_down(self):
        """Return children during the top-down exection."""
        return self.children

    # --------------------------------------------- #
    # -- Bottum-up data-flow operations (reduce) -- #
    # --------------------------------------------- #

    def bottum_up(self, store_results=True):
        if conf.DEBUG:
            debug.current = self
        # Terminaison (leaf) node return results
        if not self.get_children_bottum_up():
            return self.results
        # 1) Build sub-aggregates over children
        children_results = [child.bottum_up(store_results=False) for
            child in self.get_children_bottum_up()]
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

    def get_children_bottum_up(self):
        """Return children during the bottum-up execution."""
        return self.children

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

        recursion: boolean
            Indicates if sub-nodes (down to the leaves) and parent nodes
            (path up to the root) should be recursively loaded. Default (True).
        """
        if key is None:  # assume fs store, and point on the root of the store
            key = key_join(prot=conf.KEY_PROT_FS, path=store)
        store = get_store(key)
        loaded = store.load(key)
        node = loaded.pop(conf.STORE_NODE_PREFIX)
        node.__dict__.update(loaded)
        # Recursively load sub-tree
        if recursion and node.children:
            children = node.children
            node.children = list()
            for child in children:
                child_key = key_push(key, child)
                node.add_child(WF.load(key=child_key, recursion=recursion))
        # Recursively load nodes'path up to the root
        curr = node
        curr_key = key
        while recursion and curr.parent == '..':
            #print node.get_key()
            curr_key = key_pop(curr_key)
            parent = WF.load(key=curr_key, recursion=False)
            parent.children = list()
            parent.add_child(curr)
            #curr.parent = parent
            #parent.add_child(cu
            curr = parent
        return node

WF = WFNode
