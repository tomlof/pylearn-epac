"""
Base Workflow node plus keys manipulation utilities.

@author: edouard.duchesnay@cea.fr
@author: jinpeng.li@cea.fr
"""

import re
import sys
import ast
import numpy as np
import warnings
from abc import abstractmethod

from epac.stores import StoreMem
from epac.configuration import conf, debug
from epac.map_reduce.results import ResultSet, Result



## ================================= ##
## == Key manipulation utils      == ##
## ================================= ##

def key_push(key, basename):
    if key and basename:
        return key + conf.SEP + basename
    else:
        return key or basename


def key_pop(key, index=-1):
    """Split the key into head / tail.

    index: int
        index=-1 (default) tail = last item
        index=0  head = first item
    Example
    -------
    >>> key = 'CV/CV(nb=0)/SelectKBest/LDA'
    >>> print key_pop(key, index=-1)
    ('CV/CV(nb=0)/SelectKBest', 'LDA')
    >>> print key_pop(key, index=0)
    ('CV', 'CV(nb=0)/SelectKBest/LDA')
    """
    if index == 0:
        i = key.find(conf.SEP)
        head = key[:i]
        tail = key[(i + 1):]
    else:
        i = key.rfind(conf.SEP) + 1
        tail = key[i:]
        head = key[:(i - 1)]
    return head, tail


_match_args_re = re.compile(u'([^(]+)\(([^)]+)\)')


def key_split(key, eval=False):
    """ Split the key into signatures.

    Parameters
    ----------

    key: str

    eval_args: boolean
      If true "un-string" the parameters, and return (name, argument) tuples:
      [(item_name1, [[argname, value], ...]), ...]
    Examples
    --------
    >>> key_split(key='SelectKBest(k=1)/SVC(kernel=linear,C=1)')
    ['SelectKBest(k=1)', 'SVC(kernel=linear,C=1)']
    >>> key_split(key='SelectKBest(k=1)/SVC(kernel=linear,C=1)', eval=True)
    [[('name', 'SelectKBest'), ('k', 1)], [('name', 'SVC'),
      ('kernel', 'linear'), ('C', 1)]]
    """
    signatures = [signature for signature in key.split(conf.SEP)]
    if eval:
        return [signature_eval(signature) for signature in signatures]
    else:
        return signatures


def signature_eval(signature):
    """Evaluate the signature string, return a list of [(name, value)...]

    Parameters
    ----------
    signature: str

    Example
    -------
    >>> signature_eval('SVC(kernel=linear,C=1)')
    [('name', 'SVC'), ('kernel', 'linear'), ('C', 1)]
    """
    m = _match_args_re.findall(signature)
    if m:
        name = m[0][0]
        ret = list()
        ret.append(("name", name))
        argstr = m[0][1]
        args = [argstr.split("=") for argstr in argstr.split(",")]
        for arg in args:
            try:
                arg[1] = ast.literal_eval(arg[1])
            except ValueError:
                pass
            ret.append((arg[0], arg[1]))
        return ret
    else:
        return [("name", signature)]


## ======================================= ##
## == Workflow Node base abstract class == ##
## ======================================= ##

class BaseNode(object):
    """WorkFlow Node base abstract class"""

    def __init__(self):
        self.parent = None
        self.children = list()
        self.store = None
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

    # --------------------- #
    # -- Tree operations -- #
    # --------------------- #

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

    def add_children(self, children):
        for child in children:
            self.add_child(child)

    def get_root(self):
        """Leaves iterator"""
        curr = self
        while True:
            if not curr.parent:
                return curr
            curr = curr.parent

    def walk_nodes(self):
        """Node iterator"""
        yield self
        if self.children:
            for child in self.children:
                for yielded in child.walk_nodes():
                    yield yielded

    def walk_true_nodes(self):
        yield self
        if self.children:
            if isinstance(self.children, list):
                children = self.children
            else:
                children = [self.children[0]]
            for child in children:
                for yielded in child.walk_true_nodes():
                    yield yielded

    def walk_leaves(self):
        """Leaves iterator"""
        if not self.children or len(self.children) == 0:
            yield self
        else:
            for child in self.children:
                for yielded in child.walk_leaves():
                    yield yielded

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
        >>> from epac import CV, Grid, Pipe
        >>> from sklearn.svm import SVC
        >>> from sklearn.lda import LDA
        >>> from sklearn.feature_selection import SelectKBest
        >>> y = [1, 1, 2, 2]
        >>> wf = CV(Grid(*[Pipe(SelectKBest(k=k), SVC()) \
        ...     for k in [1, 5]]), n_folds=2, y=y)
        # List all leaves keys
        >>> for n in wf:
        ...     print n.get_key()
        ...
        CV/CV(nb=0)/Grid/SelectKBest(k=1)/SVC
        CV/CV(nb=0)/Grid/SelectKBest(k=5)/SVC
        CV/CV(nb=1)/Grid/SelectKBest(k=1)/SVC
        CV/CV(nb=1)/Grid/SelectKBest(k=5)/SVC
        # Get a single node using excat key match
        >>> wf.get_node(key="CV/CV(nb=0)/Grid/SelectKBest(k=1)").get_key()
        'CV/CV(nb=0)/Grid/SelectKBest(k=1)'
        # Get several nodes using wild cards
        >>> for n in wf.get_node(regexp="CV/*"):
        ...         print n.get_key()
        ...
        CV/CV(nb=0)
        CV/CV(nb=1)
        >>> for n in wf.get_node(regexp="*CV/CV(*)/*/*/SVC"):
        ...     print n.get_key()
        ...
        CV/CV(nb=0)/Grid/SelectKBest(k=1)/SVC
        CV/CV(nb=0)/Grid/SelectKBest(k=5)/SVC
        CV/CV(nb=1)/Grid/SelectKBest(k=1)/SVC
        CV/CV(nb=1)/Grid/SelectKBest(k=5)/SVC
        """
        if key:
            if key == self.get_key():
                return self
            for child in self.children:
                if key.find(child.get_key()) == 0:
                    return child.get_node(key)
            return None
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
        """Get path iterator from root.

        See also
        --------
        get_path_from_node(node)
        """
        return self.get_path_from_node(node=self.get_root())

    def get_path_from_node(self, node):
        """
        Get path iterator from node to self.

        Example
        -------
        >>> from epac import Perms, CV, Pipe, Methods
        >>> from sklearn.lda import LDA
        >>> from sklearn.svm import SVC
        >>> from sklearn.feature_selection import SelectKBest
        >>> root = Perms(CV(Pipe(SelectKBest(k=2), Methods(LDA(), SVC()))))
        >>> leaf = root.get_leftmost_leaf()
        >>> print leaf
        Perms/Perm(nb=0)/CV/CV(nb=0)/SelectKBest/Methods/LDA
        >>> node = root.children[2].children[0]
        >>> print node
        Perms/Perm(nb=2)/CV
        >>> print [n.get_signature() for n in leaf.get_path_from_node(node=node)]
        ['CV', 'CV(nb=0)', 'SelectKBest', 'Methods', 'LDA']
        >>> print [n.get_signature() for n in leaf.get_path_from_root()]
        ['Perms', 'Perm(nb=2)', 'CV', 'CV(nb=0)', 'SelectKBest', 'Methods', 'LDA']
        """
        if node.get_key() == self.get_key():
            yield node
        else:
            key = self.get_key()
            parent_key = node.get_key()
            trim_key = key.strip().strip(conf.SEP).strip()
            if(trim_key[0:len(parent_key)] == parent_key):
                #Remove root
                trim_key = trim_key[len(parent_key):]
                trim_key = trim_key.lstrip(conf.SEP)
            path_key = trim_key
            key_parts = key_split(path_key)
            #idx = len(key_parts) - 1
            curr = self
            # Check if node can be found in parents
            while curr and curr.get_key() != parent_key:
                curr = curr.parent
            if not curr or curr is not node:
                raise ValueError('Parent node could not be found in tree')
            # Go down from node to self
            yield curr
            while key_parts:
                signature = key_parts.pop(0)
                is_found = False
                found_pos = 0
                i = 0
                for child in curr.children:
                    if child.get_signature() == signature:
                        is_found = True
                        found_pos = i
                        break
                    i = i + 1
                if is_found:
                    curr = curr.children[found_pos]
                    yield curr

    # --------------------- #
    # -- Key             -- #
    # --------------------- #

    def get_key(self):
        """Return primary  key.

        Primary key is a unique identifier of a node in the tree, it is
        used to store leaf outputs at the end of the downstream flow.
        Intermediate key identify upstream results.
        """
        if not self.parent:
            return self.get_signature()
        elif not hasattr(self.parent, "get_key"):
            return key_push(str(self.parent), self.get_signature())
        else:
            return key_push(self.parent.get_key(), self.get_signature())

    def get_signature(self):
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

    def get_signature_args(self):
        return self.signature_args

    @abstractmethod
    def get_parameters(self):
        """Return the state of the object"""

    def get_store(self, name="default"):
        """Return the first store found on the path to tree root. If no store
        has been defined create one on the tree root and return it."""
        curr = self
        closest_store = None
        while True:
            if curr.store:
                if not closest_store:
                    closest_store = curr.store
                if curr.store.load(key_push(self.get_key(), name)):
                    return curr.store
            if not curr.parent:
                if closest_store:
                    return closest_store
                curr.store = StoreMem()
                return curr.store
            curr = curr.parent

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
        for n in self.walk_nodes():
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

    def top_down(self, **Xy):
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

        Example
        -------
        >>> from epac import Methods
        >>> from sklearn.svm import SVC
        >>> from sklearn import datasets
        >>> X, y = datasets.make_classification(n_samples=12,
        ...                                     n_features=10,
        ...                                     n_informative=2,
        ...                                     random_state=1)
        >>> methods = Methods(*[SVC(C=1), SVC(C=2)])
        >>> methods.top_down(X=X, y=y)
        [{'y/true': array([ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  1.,  0.,  1.]), 'y/pred': array([ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  1.,  0.,  1.])}, {'y/true': array([ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  1.,  0.,  1.]), 'y/pred': array([ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  1.,  0.,  1.])}]

        """
        if conf.TRACE_TOPDOWN:
            print self.get_key()
        if debug.DEBUG:
            debug.current = self
            debug.Xy = Xy
        if not self.parent:
            self.initialization(**Xy)  ## Performe some initialization
        Xy = self.transform(**Xy)
        if self.children:
            # Call children func_name down to leaves
            ret = [child.top_down(**Xy) for child in self.get_children_top_down()]
            Xy = ret[0] if len(ret) == 1 else ret
        else:
            result = Result(key=self.get_signature(), **Xy)
            self.save_results(ResultSet(result))
        return Xy

    def get_children_top_down(self):
        """Return children during the top-down exection."""
        return self.children

    run = top_down

    @abstractmethod
    def transform(self, **Xy):
        """"""

    def initialization(self, **Xy):
        conf.init_ml(**Xy)

    # --------------------------------------------- #
    # -- Bottum-up data-flow operations (reduce) -- #
    # --------------------------------------------- #
    def reduce(self, store_results=True):
        if self.children:
            # 1) Build sub-aggregates over children
            children_result_set = [child.reduce(store_results=False) for
                child in self.children]
            result_set = ResultSet(*children_result_set)
            # Append node signature in the keys
            for result in result_set:
                result["key"] = key_push(self.get_signature(), result["key"])
            return result_set
        else:
            return self.load_results()

    # -------------------------------- #
    # -- I/O persistance operations -- #
    # -------------------------------- #
    def save_results(self, results):
        """ Save ResultSet
        """
        store = self.get_store()
        store.save(key_push(self.get_key(), conf.RESULT_SET), results)

    def load_results(self):
        """ Load ResultSet
        """
        return self.get_store(name=conf.RESULT_SET).load(
                    key_push(self.get_key(), conf.RESULT_SET))

    def save_state(self, state, name="default"):
        warnings.warn("deprecated save_state", DeprecationWarning)
        store = self.get_store()
        store.save(key_push(self.get_key(), name), state)

    def load_state(self, name="default"):
        warnings.warn("deprecated load_state", DeprecationWarning)
        return self.get_store(name=name).load(key_push(self.get_key(), name))

    def save_tree(self, store):
        """I/O (persistance) operation: save the whole tree: ie.: execution
        tree + stores.

        Parameters
        ----------
        store: Store

        See Also
        --------
        Store.load()
        """
        # Save execution tree without the stores
        stores = dict()
        for node in self.walk_true_nodes():
            if node.store:
                stores[node.get_key()] = node.store
                node.store = None
        store.save(key=conf.STORE_EXECUTION_TREE_PREFIX,
                   obj=self, protocol="bin")
        for key1 in stores:
            node = self.get_node(key1)
            node.store = stores[key1]
        # Save the stores
        for key1 in stores:
            # print key1, stores[key1]
            store.save(key=key_push(key1, conf.STORE_STORE_PREFIX),
                       obj=stores[key1], protocol="bin")

    def save_node(self, store):
        """I/O (persistance) operation: save single node states ie.: store"""
        store.save(key=key_push(self.get_key(), conf.STORE_STORE_PREFIX),
                       obj=self.store, protocol="bin")

    def merge_tree_store(self, another_tree_root):
        '''Merge all the stores from another_tree_root

        Example
        -------
        >>> from epac.tests.wfexamples2test import WFExample1
        >>> from sklearn import datasets

        >>> ## Build dataset
        >>> ## =============
        >>> X, y = datasets.make_classification(n_samples=10,
        ...                                     n_features=20,
        ...                                     n_informative=5,
        ...                                     random_state=1)
        >>> Xy = {'X':X, 'y':y}
        >>> ## Build Tree and compute results
        >>> ## ==============================
        >>> tree_root_node = WFExample1().get_workflow()
        >>> tree_root_node.run(**Xy)
        [array([1, 0, 1, 0, 1, 1, 1, 0, 0, 0]), array([1, 0, 1, 0, 1, 1, 1, 0, 0, 0])]
        >>> if tree_root_node.store:
        ...     print repr(tree_root_node.store.dict)
        ... 
        {'Methods/SVC(C=1)/result_set': ResultSet(
        [{'key': SVC(C=1), 'score_te': 1.0, 'score_tr': 1.0, 'true_te': [1 0 1 0 1 1 1 0 0 0], 'pred_te': [1 0 1 0 1 1 1 0 0 0]}]), 'Methods/SVC(C=3)/result_set': ResultSet(
        [{'key': SVC(C=3), 'score_te': 1.0, 'score_tr': 1.0, 'true_te': [1 0 1 0 1 1 1 0 0 0], 'pred_te': [1 0 1 0 1 1 1 0 0 0]}])}

        >>> ## Build another tree to copy results in store
        >>> ## ===========================================
        >>> tree_root_node2 = WFExample1().get_workflow()
        >>> print tree_root_node2.store
        None
        >>> 
        >>> tree_root_node2.merge_tree_store(tree_root_node)
        >>> if tree_root_node2.store:
        ...     print repr(tree_root_node.store.dict)
        ... 
        {'Methods/SVC(C=1)/result_set': ResultSet(
        [{'key': SVC(C=1), 'score_te': 1.0, 'score_tr': 1.0, 'true_te': [1 0 1 0 1 1 1 0 0 0], 'pred_te': [1 0 1 0 1 1 1 0 0 0]}]), 'Methods/SVC(C=3)/result_set': ResultSet(
        [{'key': SVC(C=3), 'score_te': 1.0, 'score_tr': 1.0, 'true_te': [1 0 1 0 1 1 1 0 0 0], 'pred_te': [1 0 1 0 1 1 1 0 0 0]}])}
        '''
        if not self.store:
            self.store = StoreMem()
        for each_node in another_tree_root.walk_true_nodes():
            if each_node.store:
                self.store.dict.update(each_node.store.dict)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
