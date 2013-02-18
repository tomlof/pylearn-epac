"""
Epac : Embarrassingly Parallel Array Computing
"""
print __doc__

## Abreviation
## ds: downstream
## us: upstream
## tr: train
## te: test


_VERBOSE = True
_DEBUG = False

import numpy as np
from abc import abstractmethod


## =========== ##
## == Utils == ##
## =========== ##

def _list_diff(l1, l2):
    return [item for item in l1 if not item in l2]


def _list_contains(l1, l2):
    return all([item in l1 for item in l2])


def _list_union_inter_diff(*lists):
    """Return 3 lists: intersection, union and differences of lists
    """
    union = set(lists[0])
    inter = set(lists[0])
    for l in lists[1:]:
        s = set(l)
        union = union | s
        inter = inter & s
    diff = union - inter
    return list(union), list(inter), list(diff)


def _list_indices(l, val):
    return [i for i in xrange(len(l)) if l[i] == val]


def _list_of_dicts_2_dict_of_lists(list_of_dict):
    """Convert a list of dicts to a dictionnary of lists.

    Example
    -------
   >>> _list_of_dicts_2_dict_of_lists([dict(a=1, b=2), dict(a=10, b=20)])
   {'a': [1, 10], 'b': [2, 20]}
    """
    dict_of_list = dict()
    for d in list_of_dict:
        #self.children[child_idx].signature_args
        #sub_aggregate = sub_aggregates[0]
        for key2 in d.keys():
            #key2 = sub_aggregate.keys()[0]
            map_out = d[key2]
            # map_out is a dictionary
            if isinstance(map_out, dict):
                if not key2 in d.keys():
                    dict_of_list[key2] = dict()
                for key3 in map_out.keys():
                    if not key3 in dict_of_list[key2].keys():
                        dict_of_list[key2][key3] = list()
                    dict_of_list[key2][key3].append(map_out[key3])
            else:  # simply concatenate
                if not key2 in dict_of_list.keys():
                    dict_of_list[key2] = list()
                dict_of_list[key2].append(map_out)
    return dict_of_list


def _dict_diff(*dicts):
    """Find the differences in a dictionaries

    Returns
    -------
    diff_keys: a list of keys that differ amongs dicts
    diff_vals: a dict with keys values differences between dictonaries.
        If some dict differ bay keys (some keys are missing), return
        the key associated with None value

    Examples
    --------
    >>> _dict_diff(dict(a=1, b=2, c=3), dict(b=0, c=3))
    {'a': None, 'b': [0, 2]}
    """
    # Find diff in keys
    union_keys, inter_keys, diff_keys = _list_union_inter_diff(*[d.keys()
                                            for d in dicts])
    diff_vals = dict()
    for k in diff_keys:
        diff_vals[k] = None
    # Find diff in shared keys
    for k in inter_keys:
        s = set([d[k] for d in dicts])
        if len(s) > 1:
            diff_vals[k] = list(s)
    return diff_vals


def _sub_dict(d, subkeys):
    return {k: d[k] for k in subkeys}


def _as_dict(v, keys):
    """
    Ensure that v is a dict, if not create one using keys.

    Example
    -------
    >>> _as_dict(([1, 2], [3, 1]), ["x", "y"])
    {'y': [3, 1], 'x': [1, 2]}
    """
    if isinstance(v, dict):
        return v
    if len(keys) == 1:
        return {keys[0]: v}
    if len(keys) != len(v):
        raise ValueError("Do not know how to build a dictionnary with keys %s"
            % keys)
    return {keys[i]: v[i] for i in xrange(len(keys))}


def _func_get_args_names(f):
    """Return non defaults function args names
    """
    import inspect
    a = inspect.getargspec(f)
    if a.defaults:
        args_names = a.args[:(len(a.args) - len(a.defaults))]
    else:
        args_names = a.args[:len(a.args)]
    if "self" in args_names:
        args_names.remove("self")
    return args_names


## ==================== ##
## == Stores and I/O == ##
## ==================== ##

class Store(object):
    """Abstract Store"""

    def __init__(self):
        pass

    def save_map_output(key1, key2=None, val2=None, keyvals2=None):
        pass


class StoreLo(Store):
    """ Store based on Living Objects"""

    def __init__(self, storage_root):
        pass

    def save_map_output(self, key1, key2=None, val2=None, keyvals2=None):
        pass


class StoreFs(Store):
    """ Store based of file system"""

    def __init__(self):
        pass

    def key2path(self, key):
        prot, path = key_split(key)
        import os
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def save_map_output(self, key1, key2=None, val2=None, keyvals2=None):
        path = self.key2path(key1)
        import os
        if key2 and val2:
            keyvals2 = dict()
            keyvals2[key2] = val2
        for key2 in keyvals2.keys():
            val2 = keyvals2[key2]
            filename = Config.store_fs_map_output_prefix + key2 +\
                Config.store_fs_pickle_suffix
            file_path = os.path.join(path, filename)
            self.save_pickle(val2, file_path)

    def save_object(self, obj, key):
        path = self.key2path(key)
        import os
#        class_name = str(obj.__class__).split(".")[-1].\
#            replace(r"'", "").replace(r">", "")
        class_name = obj.__class__.__name__
        # try to save in json format
        filename = Config.store_fs_node_prefix + class_name +\
            Config.store_fs_json_suffix
        file_path = os.path.join(path, filename)
        if self.save_json(obj, file_path):
            # saving in json failed => pickle
            filename = Config.store_fs_node_prefix + class_name +\
            Config.store_fs_pickle_suffix
            file_path = os.path.join(path, filename)
            self.save_pickle(obj, file_path)

    def load_object(self, key):
        """Load a node given a key, recursion=True recursively walk through
        children"""
        path = self.key2path(key)
        import os
        prefix = os.path.join(path, Config.store_fs_node_prefix)
        import glob
        file_path = glob.glob(prefix + '*')
        if len(file_path) != 1:
            raise IOError('Found no or more that one file in %s*' % (prefix))
        file_path = file_path[0]
        _, ext = os.path.splitext(file_path)
        if ext == Config.store_fs_json_suffix:
            obj_dict = self.load_json(file_path)
            class_str = file_path.replace(prefix, "").\
                replace(Config.store_fs_json_suffix, "")
            obj = object.__new__(eval(class_str))
            obj.__dict__.update(obj_dict)
        elif ext == Config.store_fs_pickle_suffix:
            obj = self.load_pickle(file_path)
        else:
            raise IOError('File %s has an unkown extension: %s' %
                (file_path, ext))
        return obj

    def load_map_output(self, key):
        path = self.key2path(key)
        import os
        import glob
        map_paths = glob.glob(os.path.join(path,
            Config.store_fs_map_output_prefix) + '*')
        map_outputs = dict()
        for map_path in map_paths:
            ext = os.path.splitext(map_path)[-1]
            if ext == Config.store_fs_pickle_suffix:
                map_obj = self.load_pickle(map_path)
            if ext == Config.store_fs_json_suffix:
                map_obj = self.load_json(map_path)
            key = os.path.splitext(os.path.basename(map_path))[0].\
                replace(Config.store_fs_map_output_prefix, "", 1)
            map_outputs[key] = map_obj
        return map_outputs

    def save_pickle(self, obj, file_path):
            import pickle
            output = open(file_path, 'wb')
            pickle.dump(obj, output)
            output.close()

    def load_pickle(self, file_path):
            #u'/tmp/store/KFold-0/SVC/__node__NodeEstimator.pkl'
            import pickle
            inputf = open(file_path, 'rb')
            obj = pickle.load(inputf)
            inputf.close()
            return obj

    def save_json(self, obj, file_path):
            import json
            import os
            output = open(file_path, 'wb')
            try:
                json.dump(obj.__dict__, output)
            except TypeError:  # save in pickle
                output.close()
                os.remove(file_path)
                return 1
            output.close()
            return 0

    def load_json(self, file_path):
            import json
            inputf = open(file_path, 'rb')
            obj = json.load(inputf)
            inputf.close()
            return obj


def get_store(key):
    """ factory function returning the Store object of the class
    associated with the key parameter"""
    splits = key_split(key)
    if len(splits) != 2 and \
        not(splits[0] in (Config.key_prot_fs, Config.key_prot_lo)):
        raise ValueError('No valid storage has been associated with key: "%s"'
            % key)
    prot, path = splits
    if prot == Config.key_prot_fs:
        return StoreFs()
#    FIXME
#    elif prot == Config.key_prot_lo:
#        return StoreLo(storage_root=_Node.roots[path])
    else:
        raise ValueError("Invalid value for key: should be:" +\
        "lo for no persistence and storage on living objects or" +\
        "fs and a directory path for file system based storage")


def key_split(key):
    """Split the key in in two parts: [protocol, path]

    Example
    -------
    >>> key_split('file:///tmp/toto')
    ['file', '/tmp/toto']
    """
    return key.split(Config.key_prot_path_sep, 1)


def key_join(prot="", path=""):
    """Join protocol and path to create a key

    Example
    -------
    >>> key_join("file", "/tmp/toto")
    'file:///tmp/toto'
    """
    return prot + Config.key_prot_path_sep + path


def key_pop(key):
    return key.rsplit(Config.key_path_sep, 1)[0]


def key_push(key, basename):
    if key and basename:
        return key + Config.key_path_sep + basename
    else:
        return key or basename


def save_map_output(key1, key2=None, val2=None, keyvals2=None):
    store = get_store(key1)
    store.save_map_output(key1, key2, val2, keyvals2)


class Config:
    store_fs_pickle_suffix = ".pkl"
    store_fs_json_suffix = ".json"
    store_fs_map_output_prefix = "__map__"
    store_fs_node_prefix = "__node__"
    key_prot_lo = "mem"  # key storage protocol: living object
    key_prot_fs = "file"  # key storage protocol: file system
    key_path_sep = "/"
    key_prot_path_sep = "://"  # key storage protocol / path separator


RECURSION_UP = 1
RECURSION_DOWN = 2


class _Node(object):
    """Nodes base class"""

    def __init__(self):
        self.signature_args = None  # dict of args to build the node signature
        self.parent = None
        self.children = list()
        self.map_outputs = dict()

    def finalize_init(self, **ds_kwargs):
        """Overload this methods if init finalization is required"""
        if self.children:
            [child.finalize_init(**ds_kwargs) for child in
                self.children]

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
    # -- key can be primary or intermediate key
    # -- * Primary key is a unique identifier of a node in the tree, it is
    # --   used to store leaf outputs at the end of the downstream flow.
    # -- * Intermediate key is a unique identifier of a node in the tree.
    # --   It is used to identify 
    # --------------------- #

    def get_key(self, nb=1):
        """Return primary or intermediate key.

        All nodes contribute to primary key (key=1). Only Mapper nodes
        contribute to intermediate key (key=2)

        Argument
        --------
        nb: int
            1 (default) return the primary key
            2 return the intermediate key
        """
        if nb is 1 or (nb is 2 and isinstance(self, _NodeMapper)):
            if not self.parent:
                return self.get_signature()
            return key_push(self.parent.get_key(nb=nb), self.get_signature())
        else:
            if not self.parent:
                return ""
            return self.parent.get_key(nb=nb)

    def get_signature(self):
        """The signature of the current Node, used to build the key"""
        args_str = self.get_signature_args_str()
        args_str = "(" + args_str + ")" if args_str else ""
        return self.get_signature_name() + args_str

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

    def add_map_output(self, key=None, val=None, keyvals=None):
        """ Collect map output

        Parameters
        ----------
        key : (string) the intermediary key
        val : (dictionary, list, tuple or array) the intermediary value
        produced by the mapper.
                If key/val are provided a single map output is added

        keyvals : a dictionary of intermediary keys/values produced by the
        mapper.
        """
        if key and val:
            self.map_outputs[key] = val
        if keyvals:
            self.map_outputs.update(keyvals)

    # ------------------------------------------ #
    # -- Top-down data-flow operations (map)  -- #
    # ------------------------------------------ #

    def top_down(self, func_name, recursion=True, **ds_kwargs):
        """Top-down data processing method

            This method does nothing more that recursively call
            parent/children map. Most of time, it should be re-defined.

            Parameters
            ----------
            func_name: str
                the name of the function to be called
            recursion: boolean
                if True recursively call parent/children map. If the
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
        recursion = self.check_recursion(recursion)
        if recursion is RECURSION_UP:
            # recursively call parent map up to root
            ds_kwargs = self.parent.top_down(func_name=func_name,
                                             recursion=recursion, **ds_kwargs)
        func = getattr(self, func_name)
        ds_kwargs = func(recursion=False, **ds_kwargs)
        #ds_kwargs = self.transform(**ds_kwargs)
        if recursion is RECURSION_DOWN:
            # Call children map down to leaves
            ret = [child.top_down(func_name=func_name, recursion=recursion,
                            **ds_kwargs) for child in self.children]
            ds_kwargs = ret[0] if len(ret) == 1 else ret
        return ds_kwargs

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

    def bottum_up(self):
        # Terminaison (leaf) node return map_outputs
        if not self.children:
            return self.map_outputs
        # 1) Build sub-aggregates over children
        sub_aggregates = [child.bottum_up() for child in self.children]
        # If not reducer, act transparently
        if not isinstance(self, _NodeReducer):
            if len(sub_aggregates) == 1:
                return sub_aggregates[0]
            # if no collision in intermediary keys, merge dict and return
            merge = dict()
            [merge.update(item) for item in sub_aggregates]
            if len(merge) != np.sum([len(item) for item in sub_aggregates]):
                import warnings
                warnings.warn("Collision occured between intermediary keys. "
                "It may be due to the fact that you are running several time "
                "the same algorithm.")
            return merge
        # 2) Agregate children's sub-aggregates
        children_name, children_args  = zip(*[(child.get_signature_name(), 
                                               child.get_signature_args())
                                               for child in self.children])
        # Cheack that children have the same name, and same argument name
        # ie.: they differ only on argument values
        if len(set(children_name)) != 1:    
            raise ValueError("Children of a Reducer have different names")
        _, arg_names, diff_arg_names = _list_union_inter_diff(*[d.keys()
                                                for d in children_args])
        if diff_args_names:
            raise ValueError("Children of a Reducer have different arguements"
            "keys")
        # args_names 
#        for arg_name in args_names:

axis_names = ['kernel', 'C']

cur_arg_name = arg_names[0]
child_cur_arg_vals = [child_arg[cur_arg_name] for child_arg in children_args]
for val in set(child_arg_vals)
    children_idx = _list_indices(child_arg_vals, val)

vals = list(set([child_arg[name] for child_arg in children_args]))

for val in vals:
    
arg_names = arg_names[1:]

name = 
children_args

        aggregate = _list_of_dicts_2_dict_of_lists(sub_aggregates)


        aggregate = dict()
        for sub_aggregate in sub_aggregates:
            #self.children[child_idx].signature_args
            #sub_aggregate = sub_aggregates[0]
            for key2 in sub_aggregate.keys():
                #key2 = sub_aggregate.keys()[0]
                map_out = sub_aggregate[key2]
                # map_out is a dictionary
                if isinstance(map_out, dict):
                    if not key2 in aggregate.keys():
                        aggregate[key2] = dict()
                    for key3 in map_out.keys():
                        if not key3 in aggregate[key2].keys():
                            aggregate[key2][key3] = list()
                        aggregate[key2][key3].append(map_out[key3])
                else:  # simply concatenate
                    if not key2 in aggregate.keys():
                        aggregate[key2] = list()
                    aggregate[key2].append(map_out)
        return aggregate

    # -------------------------------- #
    # -- I/O persistance operations -- #
    # -------------------------------- #

    def save_node(self, recursion=True):
        """I/O (persistance) operation: save the node on the store"""
        key = self.get_key()
        store = get_store(key)
        # Prevent recursion saving of children/parent in a single dump:
        # replace reference to chidren/parent by basename strings
        import copy
        clone = copy.copy(self)
        clone.children = [child.get_signature() for child in self.children]
        if self.parent:
            clone.parent = ".."
        store.save_object(clone, key)
        recursion = self.check_recursion(recursion)
        if recursion is RECURSION_UP:
            # recursively call parent save up to root
            self.parent.save_node(recursion=recursion)
        if recursion is RECURSION_DOWN:
            # Call children save down to leaves
            [child.save_node(recursion=recursion) for child
                in self.children]


def load_node(key=None, store=None, recursion=True):
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
        key = key_join(prot=Config.key_prot_fs, path=store)
    store = get_store(key)
    node = store.load_object(key)
    # children contain basename string: Save the string a recursively
    # walk/load children
    recursion = node.check_recursion(recursion)
    if recursion is RECURSION_UP:
        parent_key = key_pop(key)
        parent = load_node(key=parent_key, recursion=recursion)
        parent.add_child(node)
    if recursion is RECURSION_DOWN:
        children = node.children
        node.children = list()
        for child in children:
            child_key = key_push(key, child)
            node.add_child(load_node(key=child_key, recursion=recursion))
    return node


## ================================= ##
## == Wrapper node for estimators == ##
## ================================= ##
class _NodeMapper(_Node):
    """Abstract class of _Node that contribute to transform the data"""
    def __init__(self):
        super(_NodeMapper, self).__init__()


class _NodeEstimator(_NodeMapper):
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
        if _DEBUG:
            print "-", self.get_key(), "fit, rec:", recursion
        # self.ds_kwargs = ds_kwargs # self = leaf; ds_kwargs = self.ds_kwargs
        # fit was called in a top-down recursive context
        if recursion:
            return self.top_down(func_name="fit", recursion=recursion,
                                 **ds_kwargs)
        # Regular fit
        self.estimator.fit(**_sub_dict(ds_kwargs, self.args_fit))
        if self.children:  # transform downstream data-flow (ds) for children
            return self.transform(recursion=False, **ds_kwargs)
        else:
            return self

    def transform(self, recursion=True, **ds_kwargs):
        if _DEBUG:
            print "-", self.get_key(),  "transform, rec:", recursion
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
        if _DEBUG:
            print "-", self.get_key(), "predict, rec:", recursion
        self.ds_kwargs = ds_kwargs  # self = leaf; ds_kwargs = self.ds_kwargs
        # fit was called in a top-down recursive context
        if recursion:
            return self.top_down(func_name="predict", recursion=recursion,
                                 **ds_kwargs)
        if self.children:  # if children call transform
            return self.transform(recursion=False, **ds_kwargs)
        # leaf node: do the prediction
        pred_arr = self.estimator.predict(**_sub_dict(ds_kwargs,
                                                    self.args_predict))
        pred_names = _list_diff(self.args_fit, self.args_predict)
        pred_dict = _as_dict(pred_arr, keys=pred_names)
        # If true values are provided in ds then store them
        if set(pred_names).issubset(set(ds_kwargs.keys())):
            true_dict = _sub_dict(ds_kwargs, pred_names)
            both = {"pred_" + str(k): true_dict[k] for k in pred_dict}
            both.update({"true_" + str(k): true_dict[k] for k in true_dict})
            self.add_map_output(key=self.get_key(2), val=both)
        else:  # store only predicted values
            self.add_map_output(key=self.get_key(2), val=pred_dict)
        return pred_arr


## ======================================================================== ##
## ==                                                                    == ##
## == Parallelization nodes
## ==
## ======================================================================== ##

# -------------------------------- #
# -- Reducer:  upstream data-flow                -- #
# -------------------------------- #

class _NodeReducer(_Node):
    """Abstract class of _Nodes that process (reduce) the upstream data-flow.
    data-flow"""
    def __init__(self):
        super(_NodeReducer, self).__init__()


# -------------------------------- #
# -- Splitter                   -- #
# -------------------------------- #

class _NodeSplitter(_Node):
    """Splitters are are non leaf node (degree >= 1) with children.
    They split the downstream data-flow to their children.
    They reduce the upstream data-flow from their children. Thus they are
    Reducers
    """
    def __init__(self):
        super(_NodeSplitter, self).__init__()

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


class CV(_NodeSplitter, _NodeReducer):
    """KFold CV splitter"""
    train_data_suffix = "train"
    test_data_suffix = "test"

    def __init__(self, task, n_folds, **kwargs):
        super(CV, self).__init__()
        self.n_folds = n_folds
        self.add_children([_NodeRowSlicer(signature_name="CV", nb=nb,
                               apply_on=None) for nb in xrange(n_folds)])
        for split in self.children:
            import copy
            split.add_child(_NodeFactory(copy.deepcopy(task)))
        if "y" in kwargs:
            self.finalize_init(**kwargs)

    def finalize_init(self, **ds_kwargs):
        if not "y" in ds_kwargs:
            raise KeyError("y ins not provided to finalize the initialization")
        y = ds_kwargs["y"]
        ## Classification task:  StratifiedKFold or Regressgion Kfold
        _, y_sorted = np.unique(y, return_inverse=True)
        min_labels = np.min(np.bincount(y_sorted))
        if self.n_folds <= min_labels:
            from sklearn.cross_validation import StratifiedKFold
            cv = StratifiedKFold(y=y, n_folds=self.n_folds)
        else:
            from sklearn.cross_validation import KFold
            cv = KFold(n=y.shape[0], n_folds=self.n_folds)
        nb = 0
        for train, test in cv:
            self.children[nb].set_sclices({CV.train_data_suffix: train,
                                 CV.test_data_suffix: test})
            nb += 1
        # propagate down-way
        if self.children:
            [child.finalize_init(**ds_kwargs) for child in
                self.children]

    def get_state(self):
        return dict(n_folds=self.n_folds)


class Perm(_NodeSplitter, _NodeReducer):
    """ Permutation Splitter"""

    def __init__(self, task, n_perms, permute="y", **kwargs):
        super(Perm, self).__init__()
        self.n_perms = n_perms
        self.permute = permute  # the name of the bloc to be permuted
        self.add_children([_NodeRowSlicer(signature_name="Perm", nb=nb,
                              apply_on=permute) for nb in xrange(n_perms)])
        for perm in self.children:
            import copy
            perm.add_child(_NodeFactory(copy.deepcopy(task)))
        if "y" in kwargs:
            self.finalize_init(**kwargs)

    def finalize_init(self, **ds_kwargs):
        if not "y" in ds_kwargs:
            raise KeyError("y is not provided to finalize the initialization")
        y = ds_kwargs["y"]
        from addtosklearn import Permutation
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


class MultiMethods(_NodeSplitter):
    """Parallelization is based on several reslicing of the same dataset:
    Slices can be split (shards) or a resampling of the original datasets.
    """
    def __init__(self, methods):
        super(MultiMethods, self).__init__()
        for m in methods:
            curr = _NodeFactory(m)
            self.add_child(curr)
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
#                    self.children[idx] self.signature_args
#                    import string
#                    arg_str = string.join([str(k) + "=" + \
#                        str(child_states[idx][k]) for k in diff_arg_keys], ",")
#                    self.children[idx].name = child_cls_names[idx] + "(" +\
#                        arg_str + ")"


# -------------------------------- #
# -- Slicers                    -- #
# -------------------------------- #

class _NodeSlicer(_Node):
    """ Slicers are Splitters' children, they re-sclice the downstream blocs.
    """
    def __init__(self):
        super(_NodeSlicer, self).__init__()


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
        else:
            self.slices = \
                slices.tolist() if isinstance(slices, np.ndarray) else slices

    def transform(self, recursion=True, sample_set=None, **ds_kwargs):
        if recursion:
            return self.top_down(func_name="transform", recursion=recursion,
                                 **ds_kwargs)
        data_keys = self.apply_on if self.apply_on else ds_kwargs.keys()
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
        print ds_kwargs
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


def _NodeFactory(*args, **kwargs):
    """
    _Node builder

    Positional parameters
    ---------------------
    class | (class, dict(kwargs)) | SEQ | PAR

    Keywords parameters
    -------------------
    store: string
        Load the object(s) from the given store. For file system store,
        indicates the path to the directory that contains a file prefixed by
        __node__*.

    key: string
        Load the node indexed by its key from the store. If missing then
        assume file system store and the key will point on the root of the
        store.

    Examples
    --------
        _NodeFactory(LDA())
        _NodeFactory(SVC(kernel="linear"))
        # Persistence:
        import tempfile
        store = tempfile.mktemp()
        n = _NodeFactory(SVC(), store=store)
        n.save_node()
        n2 = _NodeFactory(store=store)
    """
    node = None
    if len(args) > 0:  # Build the node from args
        ## Make it clever enough to deal single argument provided as a tuple
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = args[0]
        #import inspect
        #cls_str = args[0].__name__ if inspect.isclass(args[0]) else args[0]
        # Arg is already a _Node, then do nothing
        if len(args) == 1 and isinstance(args[0], _Node):  # _Node
            node = args[0]
        # Splitters: (KFold|StratifiedKFold|Permutation, kwargs)
        # _NodeEstimator: object
        else:
            node = _NodeEstimator(args[0])
    # store or key : load from store
    if "store" in kwargs and isinstance(kwargs["store"], str):
#        if node:
#            node.name = key_join(prot=Config.key_prot_fs, path=kwargs["store"])
#        else:
        key = kwargs["key"] if "key" in kwargs else None
        node = load_node(key=key, store=kwargs["store"])
    return(node)


def _group_args(*args):
    """group arguments provided as class, dict into a tuple"""
    args_splitted = list()
    i = 0
    while i < len(args):
        if isinstance(args[i], _Node):                           # _Node
            args_splitted.append(args[i])
            i += 1
        elif i + 1 < len(args) and isinstance(args[i + 1], dict): # class, dict
            args_splitted.append((args[i], args[i + 1]))
            i += 2
        else:
            args_splitted.append(args[i])                      # class or obj
            i += 1
    return args_splitted


def Seq(*args):
    """
    Parameters
    ----------
    TASK [, TASK]*

    Examples
    --------
        SEQ((SelectKBest, dict(k=2)),  (SVC, dict(kernel="linear")))
    """
    # SEQ(_Node [, _Node]*)
    args = _group_args(*args)
    root = None
    for task in args:
        curr = _NodeFactory(task)
        if not root:
            root = curr
        else:
            prev.add_child(curr)
        prev = curr
    return root


def Par(*args, **kwargs):
    """
    Syntax (positional parameters)
    ------------------------------


    Keywords parameters
    -------------------
    data: dict
        Use
    store: string
        Store (recursively) the objects tree on the given store.
        For file system store, indicates a path to a directory.

    Examples
    --------
    """
    args = _group_args(*args)
    root = MultiMethods(methods=args)
    # if data is provided finalize the initialization
    if "finalize" in kwargs:
        root.finalize_init(**kwargs["finalize"])
    if "store" in kwargs and isinstance(kwargs["store"], str):
        root.name = key_join(prot=Config.key_prot_fs, path=kwargs["store"])
        root.save_node()
    return root
