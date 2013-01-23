"""
Epac : Embarrassingly Parallel Array Computing
"""
print __doc__

## Abreviation
## ds: downstream
## us: upstream
## tr: train
## te: test

import numpy as np
#from abc import abstractmethod

## =========== ##
## == Utils == ##
## =========== ##

def _sub_dict(d, subkeys):
    return {k:d[k] for k in subkeys}

def _sub_dict_set(d, new_vals, subkeys=None):
    """ Set d[subkeys] to new_vals

    Arguments
    ---------
    d: dict
    
    subkeys: list of (sub) keys, if missing use new_vals.keys()

    new_vals: singleton, tuple, dict
        if singleton convert it to a tuple of length one
        if tuple convert it to a dict using "subkeys" keys        
    """
    if not subkeys:
        subkeys = new_vals.keys()
    if not _list_contains(d.keys(), subkeys):
        raise ValueError('Some keys of subkeys are not in d.keys()')
    d = d.copy() # avoid side effect
    # if singleton (and not dict) convert to length one tuple
    if not isinstance(new_vals, (tuple, dict)):
        new_vals = (new_vals,)
    # if tuple convert to dict, with keys matching subkeys
    if isinstance(new_vals, tuple): # transform list to dict, matching order
        if len(subkeys) is not len(new_vals):
            raise ValueError('Arguments of different lengths')
        new_vals = {subkeys[i]:new_vals[i] for i in xrange(len(new_vals))}
    # Now new_vals is a dict, replace values in d
    for k in new_vals.keys():
        d[k] = new_vals[k]
    return d

def _list_diff(l1, l2):
    return [item for item in l1 if not item in l2]

def _list_contains(l1, l2):
    return all([item in l1 for item in l2])


def _get_args_names(f):
    import inspect
    a=inspect.getargspec(f)
    if a.defaults:
        args_names = a.args[:(len(a.args)-len(a.defaults))]
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
        class_name = str(obj.__class__).split(".")[-1].\
            replace(r"'", "").replace(r">", "")
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
    elif prot == Config.key_prot_lo:
        return StoreLo(storage_root=Node.roots[path])
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
    ds_kwargs_data_prefix = ["X", "y"]


RECURSION_UP = 1
RECURSION_DOWN = 2


class Node(object):
    """Parallelization node, provide:
        - key/val
        - I/O interface with the store."""

    def __init__(self, name=None, key=None, store=None):
        """
        Parameter
        ---------
        name: the node's name, used to build its key
        """
        self.name = name
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

    def get_name(self):
        return self.name

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
        if nb is 1 or (nb is 2 and isinstance(self, NodeMapper)):
            if not self.parent:
                return self.get_name()
            return key_push(self.parent.get_key(nb=nb), self.get_name())
        else:
            if not self.parent:
                return ""
            return self.parent.get_key(nb=nb)

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

    def top_down(self, recursion=True, **ds_kwargs):
        """Top-down data processing method

            This method does nothing more that recursively call
            parent/children map. Most of time, it should be re-defined.

            Parameters
            ----------
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
        print "top_down", self.get_key()
        recursion = self.check_recursion(recursion)
        if recursion is RECURSION_UP:
            # recursively call parent map up to root
            ds_kwargs = self.parent.top_down(recursion=recursion,
                                                     **ds_kwargs)
        ds_kwargs = self.transform(**ds_kwargs)
        if recursion is RECURSION_DOWN:
            # Call children map down to leaves
            [child.top_down(recursion=recursion, **ds_kwargs)
                for child in self.children]
        return ds_kwargs

    def transform(self, **kwargs):
        return kwargs

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
        # Terminaison (leaf) node
        if not self.children:
            return self.map_outputs
        # 1) Build sub-aggregates over children
        sub_aggregates = [child.bottum_up() for child in self.children]
        # If not reducer, act transparently
        if not isinstance(self, NodeReducer):
            if len(sub_aggregates) == 1:
                return sub_aggregates[0]
            # if no collision in intermediary keys, merge dict and return
            merge = dict()
            [merge.update(item) for item in sub_aggregates]
            if len(merge) != np.sum([len(item) for item in sub_aggregates]):
                ValueError("Collision occured between intermediary keys")
            return merge
        # 2) Agregate children's sub-aggregates
        aggregate = dict()
        for sub_aggregate in sub_aggregates:
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
        clone.children = [child.name for child in self.children]
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
class NodeMapper(Node):
    """Abstract class of Node that contribute to transform the data"""
    def __init__(self, name):
        super(NodeMapper, self).__init__(name=name)


class NodeEstimator(NodeMapper):
    """Node that wrap estimators"""

    def __init__(self, estimator):
        self.estimator = estimator
        super(NodeEstimator, self).__init__(
            name=estimator.__class__.__name__)
        self.args_fit = _get_args_names(self.estimator.fit) \
            if hasattr(self.estimator, "fit") else None
        self.args_predict = _get_args_names(self.estimator.predict) \
            if hasattr(self.estimator, "predict") else None
        self.args_transform = _get_args_names(self.estimator.transform) \
            if hasattr(self.estimator, "transform") else None

    def __repr__(self):
        return '%s(estimator=%s)' % (self.__class__.__name__,
            self.estimator.__repr__())

    def transform(self, **ds_kwargs):
        self.ds_kwargs = ds_kwargs # self = leaf; ds_kwargs = self.ds_kwargs
        ds_kwargs_tr, ds_kwargs_te = NodeKFold.split_tr_te(**ds_kwargs)
        # Fit the training data selecting only args_fit in stream
        self.estimator.fit(**_sub_dict(ds_kwargs_tr, self.args_fit))
        if self.children: # transform downstream (ds)
            # train
            new_vals = self.estimator.transform(**_sub_dict(ds_kwargs_tr,
                                                 self.args_transform))
            ds_kwargs_tr = _sub_dict_set(d=ds_kwargs_tr,
                new_vals=new_vals, subkeys=self.args_transform)
            # test
            new_vals = self.estimator.transform(**_sub_dict(ds_kwargs_te,
                                                 self.args_transform))
            ds_kwargs_te = _sub_dict_set(d=ds_kwargs_te,
                new_vals=new_vals, subkeys=self.args_transform)
            # join train, test into downstream
            ds_kwargs = NodeKFold.join_tr_te(ds_kwargs_tr, ds_kwargs_te)
            return ds_kwargs
        else:
            # leaf node: do the prediction predict the test
            # get args in fit but not in predict
            args_predicted = _list_diff(self.args_fit, self.args_predict)
            true = _sub_dict(ds_kwargs_te, args_predicted)
            # predict test downstream
            pred = self.estimator.predict(
                **_sub_dict(ds_kwargs_te, self.args_predict))
            pred = _sub_dict_set(true, new_vals=pred, subkeys=args_predicted)
            both = {k+"_true":true[k] for k in true}
            for k in pred:
                both[k+"_pred"]=pred[k]
            # collect map output
            self.add_map_output(key=self.get_key(2), val=both)
            return both



## =========================== ##
## == Parallelization nodes == ##
## =========================== ##

class NodeReducer(Node):
    """Abstract class of Node that contribute to redcue the data"""
    def __init__(self, name):
        super(NodeReducer, self).__init__(name=name)

# -------------------------------- #
# -- Slicers                    -- #
# -------------------------------- #

class NodeSlicer(Node):
    """ Slicers are Splitters' children, they re-sclice the downstream blocs.
    """
    def __init__(self, name):
        super(NodeSlicer, self).__init__(name=name)


class NodeRowSlicer(NodeSlicer):
    """Row-wise reslicing of the downstream blocs.

    Parameters
    ----------
    name: string
    
    apply_on: string or list of strings
        The name(s) of the downstream blocs to be rescliced. If
        None, all downstream blocs are rescliced.
    """

    def __init__(self, name, apply_on):
        super(NodeRowSlicer, self).__init__(name=name)
        self.slices = None
        self.apply_on = apply_on

    def finalize_init(self, **ds_kwargs):
        ds_kwargs = self.transform(**ds_kwargs)
        # print self, "(",self.parent,")", self.slices, ds_kwargs
        # propagate down-way
        if self.children:
            [child.finalize_init(**ds_kwargs) for child in
                self.children]
                
    def set_sclices(self, slices):
        # convert as a list if required
        if isinstance(slices, dict):
            self.slices =\
                {k: slices[k].tolist() if isinstance(slices[k], np.ndarray)
                else slices[k] for k in slices}
        else:
            self.slices = \
                slices.tolist() if isinstance(slices, np.ndarray) else slices

    def transform(self, **ds_kwargs):
        keys_data = self.apply_on if self.apply_on else ds_kwargs.keys()
        data_out = ds_kwargs.copy()
        for key_data in keys_data:  # slice input data
            if isinstance(self.slices, dict):
                # rename output keys according to input keys and slice keys
                data = data_out.pop(key_data)
                for key_slice in self.slices:
                    data_out[key_data + key_slice] = \
                        data[self.slices[key_slice]]
            else:
                data_out[key_data] = data_out[key_data][self.slices]
        return data_out


# -------------------------------- #
# -- Splitter                   -- #
# -------------------------------- #

class NodeSplitter(Node):
    """Splitters"""
    def __init__(self, name):
        super(NodeSplitter, self).__init__(name=name)

class NodeKFold(NodeSplitter, NodeReducer):
    """KFold splitter"""
    train_data_suffix = "train"
    test_data_suffix = "test"

    def __init__(self, n, n_folds):
        super(NodeKFold, self).__init__(name="KFold")
        self.n = n
        self.n_folds = n_folds
        self.add_children([NodeRowSlicer(name=str(nb), apply_on=None) for nb\
                in xrange(n_folds)])
                
    def finalize_init(self, **ds_kwargs):
        if isinstance(self.n, str): # self.n need to be evaluated
            self.n = eval(self.n, ds_kwargs.copy())
        from sklearn.cross_validation import KFold
        nb = 0
        for train, test in KFold(n=self.n, n_folds=self.n_folds):
            self.children[nb].set_sclices({NodeKFold.train_data_suffix: train,
                                 NodeKFold.test_data_suffix: test})
            self.children[nb].n = self.n
            nb += 1
        # propagate down-way
        if self.children:
            [child.finalize_init(**ds_kwargs) for child in
                self.children]

    @classmethod
    def split_tr_te(cls, **ds_kwargs):
        """Split ds_kwargs into train dict (that contains train suffix in kw)
        and test dict (that contains test suffix in kw).

        Returns
        -------
        Two dictionaries without the train and test suffix into kw. Outputs
        are then compliant with estimator API that takes only X, y paramaters.
        If only "X" an "y" kw are found they are replicated into the both
        outputs.

        Example
        -------
       >>> NodeKFold.split_tr_te(Xtrain=1, ytrain=2, Xtest=-1, ytest=-2,
       ... a=33)
       ({'y': 2, 'X': 1, 'a': 33}, {'y': -2, 'X': -1, 'a': 33})

        >>> NodeKFold.split_tr_te(X=1, y=2, a=33)
        ({'y': 2, 'X': 1, 'a': 33}, {'y': 2, 'X': 1, 'a': 33})
        """
        ds_kwargs_tr = ds_kwargs.copy()
        ds_kwargs_te = ds_kwargs.copy()
        for data_key in Config.ds_kwargs_data_prefix:
            data_key_tr = data_key + NodeKFold.train_data_suffix
            data_key_te = data_key + NodeKFold.test_data_suffix
            # Remove [X|y]test from train
            if data_key_te in ds_kwargs_tr:
                ds_kwargs_tr.pop(data_key_te)
            # Remove [X|y]train from test
            if data_key_tr in ds_kwargs_te:
                ds_kwargs_te.pop(data_key_tr)
            # [X|y]train becomes [X|y] to be complient with estimator API
            if data_key_tr in ds_kwargs_tr:
                ds_kwargs_tr[data_key] =\
                    ds_kwargs_tr.pop(data_key_tr)
            # [X|y]test becomes [X|y] to be complient with estimator API
            if data_key_te in ds_kwargs_te:
                ds_kwargs_te[data_key] =\
                    ds_kwargs_te.pop(data_key_te)
        return ds_kwargs_tr, ds_kwargs_te

    @classmethod
    def join_tr_te(cls, ds_kwargs_tr, ds_kwargs_te):
        """Merge train test separate ds_kwargs into single dict.

            Returns
            -------
            A single dictionary with train and test suffix as kw.

            Example
            -------
            >>> NodeKFold.join_tr_te(dict(X=1, y=2, a=33),
            ...                          dict(X=-1, y=-2, a=33))
            {'ytest': -2, 'Xtest': -1, 'a': 33, 'Xtrain': 1, 'ytrain': 2}
        """
        ds_kwargs = dict()
        for data_key in Config.ds_kwargs_data_prefix:
            if data_key in ds_kwargs_tr:  # Get [X|y] from train
                data_key_tr = data_key + NodeKFold.train_data_suffix
                ds_kwargs[data_key_tr] = \
                    ds_kwargs_tr.pop(data_key)
            if data_key in ds_kwargs_te:  # Get [X|y] from train
                data_key_te = data_key + NodeKFold.test_data_suffix
                ds_kwargs[data_key_te] = \
                    ds_kwargs_te.pop(data_key)
        ds_kwargs.update(ds_kwargs_tr)
        ds_kwargs.update(ds_kwargs_te)
        return ds_kwargs

class NodeStratifiedKFold(NodeSplitter, NodeReducer):
    """ StratifiedKFold Splitter"""

    def __init__(self, y, n_folds):
        super(NodeStratifiedKFold, self).__init__(name="StratifiedKFold")
        self.y = y
        self.n_folds = n_folds
        self.add_children([NodeRowSlicer(name=str(nb), apply_on=None) for nb\
                in xrange(n_folds)])
                
    def finalize_init(self, **ds_kwargs):
        if isinstance(self.y, str): # self.y need to be evaluated
            self.y = eval(self.y, ds_kwargs.copy())
        from sklearn.cross_validation import StratifiedKFold
        nb = 0
        for train, test in StratifiedKFold(y=self.y, n_folds=self.n_folds):
            self.children[nb].set_sclices({NodeKFold.train_data_suffix: train,
                                 NodeKFold.test_data_suffix: test})
            self.children[nb].y = self.y
            nb += 1
        # propagate down-way
        if self.children:
            [child.finalize_init(**ds_kwargs) for child in
                self.children]


class NodePermutation(NodeSplitter, NodeReducer):
    """ Permutation Splitter"""

    def __init__(self, n, n_perms, apply_on):
        super(NodePermutation, self).__init__(name="Permutation")
        self.n = n
        self.n_perms = n_perms
        self.appy_on = apply_on
        self.add_children([NodeRowSlicer(name=str(nb), apply_on=apply_on) \
            for nb in xrange(n_perms)])

    def finalize_init(self, **ds_kwargs):
        if isinstance(self.n, str): # self.n need to be evaluated
            self.n = eval(self.n, ds_kwargs.copy())
        from addtosklearn import Permutation
        nb = 0
        for perm in Permutation(n=self.n, n_perms=self.n_perms):
            self.children[nb].set_sclices(perm)
            self.children[nb].n = self.n
            nb += 1
        # propagate down-way
        if self.children:
            [child.finalize_init(**ds_kwargs) for child in
                self.children]

class MultiMethods(NodeSplitter):
    """Parallelization is based on several reslicing of the same dataset:
    Slices can be split (shards) or a resampling of the original datasets.
    """
    def __init__(self):
        super(MultiMethods, self).__init__(name="MultiMethods")
        
def NodeFactory(*args, **kwargs):
    """
    Node builder

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
        NodeFactory(LDA)
        NodeFactory(SVC(kernel="linear"))
        NodeFactory(KFold, dict(n=10, n_folds=4))
        # Persistence:
        import tempfile
        store = tempfile.mktemp()
        n = NodeFactory(SVC(), store=store)
        n.save_node()
        n2 = NodeFactory(store=store)
    """
    node = None
    if len(args) > 0: # Build the node from args 
        ## Make it clever enough to deal single argument provided as a tuple
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = args[0]
        import inspect
        cls_str = args[0].__name__ if inspect.isclass(args[0]) else args[0]
        # Arg is already a Node, then do nothing
        if len(args) == 1 and isinstance(args[0], Node):  # Node
            node = args[0]        
        # Splitters: (KFold|StratifiedKFold|Permutation, kwargs)
        elif cls_str in ("KFold", "StratifiedKFold", "Permutation")\
            and len(args) == 2:
            kwargs = args[1]
            if cls_str == "KFold":
                node = NodeKFold(**kwargs)
            elif cls_str == "StratifiedKFold":
                node = NodeStratifiedKFold(**kwargs)
            elif cls_str == "Permutation":
                node = NodePermutation(**kwargs)
        # NodeEstimator: class|class, kwargs
        elif inspect.isclass(args[0]):
            instance = object.__new__(args[0])
            if len(args) == 1:          # class 
                node = NodeEstimator(instance.__init__())
            else:                       # class, kwargs
                node = NodeEstimator(instance.__init__(**args[1]))
        # NodeEstimator: object
        else:
             node = NodeEstimator(args[0])
    # store or key : load from store
    if "store" in kwargs and isinstance(kwargs["store"], str):
        if node:
            node.name = key_join(prot=Config.key_prot_fs, path=kwargs["store"])
        else:
            key = kwargs["key"] if "key" in kwargs else None
            node = load_node(key=key, store=kwargs["store"])
    return(node)


def _group_args(*args):
    """group arguments provided as class, dict into a tuple"""
    args_splitted = list()
    i = 0
    while i < len(args):
        if isinstance(args[i], Node):                           # Node
            args_splitted.append(args[i])
            i += 1
        elif i + 1 < len(args) and isinstance(args[i+1], dict): # class, dict
            args_splitted.append((args[i], args[i+1]))
            i += 2
        else:
            args_splitted.append(args[i])                      # class or obj
            i += 1
    return args_splitted

def SEQ(*args):
    """    
    Parameters
    ----------
    TASK [, TASK]*

    Examples
    --------
        SEQ((SelectKBest, dict(k=2)),  (SVC, dict(kernel="linear")))
    """
    # SEQ(Node [, Node]*)
    args = _group_args(*args)
    root = None
    for task in args:
        curr = NodeFactory(task)
        if not root:
            root = curr
        else:
            prev.add_child(curr)
        prev = curr
    return root


def PAR(*args, **kwargs):
    """
    Syntax (positional parameters)
    ------------------------------
    PAR     ::= PAR(Node [, Node]+)
            ::= PAR(Splitter, Node)
    Splitter::= KFold|StratifiedKFold|Permutation, kwargs

 
    Keywords parameters
    -------------------
    data: dict
        Use 
    store: string
        Store (recursively) the objects tree on the given store. For file system 
        store, indicates a path to a directory.

    Examples
    --------
        from sklearn.cross_validation import KFold, StratifiedKFold
        from sklearn.feature_selection import SelectKBest
        from addtosklearn import Permutation
        from sklearn.svm import SVC
        from sklearn.lda import LDA        
        PAR(LDA(),  SVC(kernel="linear"))
        PAR(KFold, dict(n="y.shape[0]", n_folds=3), LDA())
        # 2 permutations of 3 folds of univariate filtering of SVM and LDA
        import tempfile
        import numpy as np
        store = tempfile.mktemp()
        X = np.asarray([[1, 2], [3, 4], [5, 6], [7, 8], [-1, -2], [-3, -4], [-5, -6], [-7, -8]])
        y = np.asarray([1, 1, 1, 1, -1, -1, -1, -1])
        
        # Design of the exectuion tree
        anova_svm = SEQ(SelectKBest(k=2), SVC(kernel="linear"))
        anova_lda = SEQ(SelectKBest(k=2), LDA())
        algos = PAR(anova_svm, anova_lda)
        algos_cv = PAR(StratifiedKFold, dict(y="y", n_folds=2), algos)
        perms = PAR(Permutation, dict(n="y.shape[0]", n_perms=3), algos_cv,
                   finalize=dict(y=y), store=store)
        perms2 = NodeFactory(store=store)
        # run
        [leaf.top_down(X=X, y=y) for leaf in perms2]
    """
    args = _group_args(*args)
    first = NodeFactory(args[0])
    # PAR(Splitter, Node)
    if isinstance(first, NodeSplitter):
        # first is a splitter ie.: a list of nodes
        task = args[1]
        for split in first.children:
            import copy
            split.add_child(NodeFactory(copy.deepcopy(task)))
        root = first
    else: # PAR(Node [, Node]+)
        root = MultiMethods()
        root.add_child(first)
        for task in args[1:]:
            curr = NodeFactory(task)
            root.add_child(curr)
    # if data is provided finalize the initialization
    if "finalize" in kwargs:
        root.finalize_init(**kwargs["finalize"])
    if "store" in kwargs and isinstance(kwargs["store"], str):
        root.name = key_join(prot=Config.key_prot_fs, path=kwargs["store"])
        root.save_node()
    return root

