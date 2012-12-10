"""
epac : Embarrassingly Parallel Array Computing
"""
print __doc__


import numpy as np


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
            filename = Epac.config.store_fs_map_output_prefix + key2 +\
                Epac.config.store_fs_pickle_suffix
            file_path = os.path.join(path, filename)
            self.save_pickle(val2, file_path)

    def save_object(self, obj, key):
        path = self.key2path(key)
        import os
        class_name = str(obj.__class__).split(".")[-1].\
            replace(r"'", "").replace(r">", "")
        # try to save in json format
        filename = Epac.config.store_fs_node_prefix + class_name +\
            Epac.config.store_fs_json_suffix
        file_path = os.path.join(path, filename)
        if self.save_json(obj, file_path):
            # saving in json failed => pickle
            filename = Epac.config.store_fs_node_prefix + class_name +\
            Epac.config.store_fs_pickle_suffix
            file_path = os.path.join(path, filename)
            self.save_pickle(obj, file_path)

    def load_object(self, key):
        """Load a node given a key, recursive=True recursively walk through
        children"""
        path = self.key2path(key)
        import os
        prefix = os.path.join(path, Epac.config.store_fs_node_prefix)
        import glob
        file_path = glob.glob(prefix + '*')
        if len(file_path) != 1:
            raise IOError('Found no or more that one file in %s' % (prefix))
        file_path = file_path[0]
        _, ext = os.path.splitext(file_path)
        if ext == Epac.config.store_fs_json_suffix:
            obj_dict = self.load_json(file_path)
            class_str = file_path.replace(prefix, "").\
                replace(Epac.config.store_fs_json_suffix, "")
            obj = object.__new__(eval(class_str))
            obj.__dict__.update(obj_dict)
        elif ext == Epac.config.store_fs_pickle_suffix:
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
            Epac.config.store_fs_map_output_prefix) + '*')
        map_outputs = dict()
        for map_path in map_paths:
            ext = os.path.splitext(map_path)[-1]
            if ext == Epac.config.store_fs_pickle_suffix:
                map_obj = self.load_pickle(map_path)
            if ext == Epac.config.store_fs_json_suffix:
                map_obj = self.load_json(map_path)
            key = os.path.splitext(os.path.basename(map_path))[0].\
                replace(Epac.config.store_fs_map_output_prefix, "", 1)
            map_outputs[key] = map_obj
        return map_outputs

    def save_pickle(self, obj, file_path):
            import pickle
            output = open(file_path, 'wb')
            pickle.dump(obj, output)
            output.close()

    def load_pickle(self, file_path):
            #u'/tmp/store/KFold-0/SVC/__node__WrapEstimator.pkl'
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
    prot, path = key_split(key)
    if prot == Epac.config.key_prot_fs:
        return StoreFs()
    elif prot == Epac.config.key_prot_lo:
        return StoreLo(storage_root=Epac.roots[path])
    else:
        raise ValueError("Invalid value for key: should be:" +\
        "lo for no persistence and storage on living objects or" +\
        "fs and a directory path for file system based storage")


def key_split(key):
    return key.split(Epac.config.key_prot_path_sep, 1)


def key_join(prot="", path=""):
    return prot + Epac.config.key_prot_path_sep + path


def key_pop(key):
    import os
    return os.path.dirname(key)


def key_push(key, basename):
    return key + Epac.config.key_path_sep + basename


def save_map_output(key1, key2=None, val2=None, keyvals2=None):
    store = get_store(key1)
    store.save_map_output(key1, key2, val2, keyvals2)


class Epac(object):
    """Parallelization node, provide:
        - key/val
        - I/O interface with the store."""

    # Static fields: config
    class config:
        store_fs_pickle_suffix = ".pkl"
        store_fs_json_suffix = ".json"
        store_fs_map_output_prefix = "__map__"
        store_fs_node_prefix = "__node__"
        key_prot_lo = "mem"  # key storage protocol: living object
        key_prot_fs = "file"  # key storage protocol: file system
        key_path_sep = "/"
        key_prot_path_sep = "://"  # key storage protocol / path separator
        kwargs_data_prefix = ["X", "y"]
        recursive_up = 1
        recursive_down = 2

    def __init__(self, steps=None, key=None, store=None, **kwargs):
        self.__dict__.update(kwargs)
        self.parent = None
        self.children = list()
        self.map_outputs = dict()
        # If a steps is provided: initial construction of the execution tree
        if steps:
            if not store:
                import string
                import random
                self.name = key_join(prot=Epac.config.key_prot_lo,
                    path="".join(random.choice(string.ascii_uppercase +
                        string.digits) for x in range(10)))
                self.build_tree(steps, **kwargs)
            # store is a string and a valid directory , assume that storage
            # will be done on the file system, ie.: key prefix "fs://"
            elif isinstance(store, str):
                self.name = key_join(prot=Epac.config.key_prot_fs,
                                     path=store)
                self.build_tree(steps, **kwargs)
                self.save_node()
            else:
                raise ValueError("Invalid value for store: should be: " +\
                "None for no persistence and storage on living objects or " +\
                "a string path for file system based storage")
        # If not steps but store or key : load from fs store
        if not steps and (isinstance(store, str) or isinstance(key, str)):
            root = load_node(key=key, store=store)
            self.__dict__.update(root.__dict__)

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

    def get_key(self):
        if not self.parent:
            return self.get_name()
        return key_push(self.parent.get_key(), self.get_name())

    def get_leaves(self):
        if not len(self.children):
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

    def build_tree(self, steps, **kwargs):
        """Build execution tree.

        Parameters
        ----------
        steps: list
        """
        if len(steps) == 0:
            return
        # If current step is a Parallelization node: a foactory of ParNode
        if isinstance(steps[0], ParNodeFactory):
            for child in steps[0].produceParNodes():
                self.add_children(child)
                child.build_tree(steps[1:], **kwargs)
        else:
            child = WrapEstimator(steps[0])
            self.add_children(child)
            child.build_tree(steps[1:], **kwargs)

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

    def check_recursive(self, recursive):
        print self.get_key(), recursive
        if recursive and type(recursive) is bool:
            if not self.children:
                return Epac.config.recursive_up
            if not self.parent:
                return Epac.config.recursive_down
            raise ValueError('recursive is True, but the node is neither a \
            leaf or the root tree, it then not possible to guess \
            if recurssion should go up or down')
        return recursive

    # ------------------------------------------ #
    # -- Top-down data-flow operations (map)  -- #
    # ------------------------------------------ #

    def map(self, recursive=True, **kwargs):
        """Top-down data processing method

            This method does nothing more that recursively call
            parent/children map. Most of time, it should be re-defined.

            Parameters
            ----------
            recursive: boolean
                if True recursively call parent/children map. If the
                current node is the root of the tree call the children.
                This way the whole tree is executed.
                If it is a leaf, then recursively call the parent before
                being executed. This a pipeline made of the path from the
                leaf to the root is executed.
            **kwargs: dict
                the keyword dictionnary of data flow

            Return
            ------
            A dictionnary of processed data
        """
        recursive = self.check_recursive(recursive)
        if recursive is Epac.config.recursive_up:
            # Recursively call parent map up to root
            kwargs = self.parent.map(recursive=recursive, **kwargs)
        if recursive is Epac.config.recursive_down:
            # Call children map down to leaves
            [child.map(recursive=recursive, **kwargs) for child
                in self.children]
        return kwargs

    # --------------------------------------------- #
    # -- Bottum-up data-flow operations (reduce) -- #
    # --------------------------------------------- #

    def reduce(self):
        # Terminaison (leaf) node
        if len(self.children) == 0:
            return self.map_outputs
        # 1) Build sub-aggregates over children
        sub_aggregates = [child.reduce() for child in self.children]
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

    def save_node(self, recursive=True):
        """I/O (persistance) operation: save the node on the store"""
        key = self.get_key()
        store = get_store(key)
        # Prevent recursive saving of children/parent in a single dump:
        # replace reference to chidren/parent by basename strings
        import copy
        clone = copy.copy(self)
        clone.children = [child.name for child in self.children]
        if self.parent:
            clone.parent = ".."
        store.save_object(clone, key)
        if recursive and len(self.children):
            for child in self.children:
                child.save_node(recursive=True)


def load_node(key=None, store=None, recursive=True):
    """I/O (persistance) operation load a node from the store"""
    if key is None:
        key = key_join(prot=Epac.config.key_prot_fs, path=store)
    #self.add_children(self.build_execution_tree(steps, data))
    store = get_store(key)
    node = store.load_object(key)
    # children contain basename string: Save the string a Recursively
    # walk/load children
    children = node.children
    node.children = list()
    if recursive and len(children):
        for child in children:
            child_key = key_push(key, child)
            node.add_child(load_node(key=child_key, recursive=True))
    return node


## ================================= ##
## == Wrapper node for estimators == ##
## ================================= ##

class WrapEstimator(Epac):
    """Node that wrap estimators"""

    def __init__(self, estimator, **kwargs):
        self.estimator = estimator
        super(WrapEstimator, self).__init__(
            name=estimator.__class__.__name__, **kwargs)

    def __repr__(self):
        return '%s(estimator=%s)' % (self.__class__.__name__,
            self.estimator.__repr__())

    def map(self, recursive=True, **kwargs):
        recursive = self.check_recursive(recursive)
        if recursive is Epac.config.recursive_up:
            # Should parent map being recursively be called before the
            # current one?
            kwargs = self.parent.map(rec_up=True, **kwargs)
        kwargs_train, kwargs_test = ParKFold.split_train_test(**kwargs)
        self.estimator.fit(**kwargs_train)            # fit the training data
        if self.children:                         # transform input to output
            kwargs_train_out = self.estimator.transform(**kwargs_train)
            kwargs_test_out = self.estimator.transform(**kwargs_test)
            kwargs_out = ParKFold.join_train_test(kwargs_train_out,
                                                  kwargs_test_out)
            # Sould children map being recursively be called after the
            # current node?
            if recursive is Epac.config.recursive_down:
                [child.map(rec_down=True, **kwargs) for child in self.children]
            else:
                return kwargs_out
        else:                 # leaf node: do the prediction predict the test
            y_true = kwargs_test.pop("y")
            y_pred = self.estimator.predict(**kwargs_test)
            out = dict(y_true=y_true, y_pred=y_pred)
            self.add_map_output(keyvals=out)             # collect map output
            return out

    #def reduce(self, recursive=True, **kwargs):
    #    pass


## =========================== ##
## == Parallelization nodes == ##
## =========================== ##

class ParNodeFactory(object):
    """Abstract class for Factories of parallelization nodes that implement
    produceParNodes"""

    def produceParNodes(self):
        raise NotImplementedError("Cannot call abstract method")


class ParSlicer(Epac):
    """Parallelization is based on several reslicing of the same dataset:
    Slices can be split (shards) or a resampling of the original datasets.
    """
    def __init__(self, transform_only=None, **kwargs):
        super(ParSlicer, self).__init__(**kwargs)
        self.transform_only = transform_only


class ParRowSlicer(ParSlicer):
    """Parallelization is based on several row-wise reslicing of the same
    dataset

    Parameters
    ----------
    slices: dict of sets of slicing indexes or a single set of slicing indexes
    """

    def __init__(self, slices, **kwargs):
        super(ParRowSlicer, self).__init__(**kwargs)
        # convert a as list if required
        if slices and  isinstance(slices, dict):
            self.slices =\
                {k: slices[k].tolist() if isinstance(slices[k], np.ndarray)
                else slices[k] for k in slices}
        else:
            self.slices = \
                slices.tolist() if isinstance(slices, np.ndarray) else slices

    def map(self, recursive=True, **kwargs):
        """ Transform inputs kwargs of array, and produce dict of array"""
        recursive = self.check_recursive(recursive)
        # Should parent map be recursively be called before the current one ?
        if recursive is Epac.config.recursive_up:
            kwargs = self.parent.map(rec_up=True, **kwargs)
        keys_data = self.transform_only if self.transform_only\
                    else kwargs.keys()
        data_out = kwargs.copy()
        for key_data in keys_data:  # slice input data
            if isinstance(self.slices, dict):
                # rename output keys according to input keys and slice keys
                data = data_out.pop(key_data)
                for key_slice in self.slices:
                    data_out[key_data + key_slice] = \
                        data[self.slices[key_slice]]
            else:
                data_out[key_data] = data_out[key_data][self.slices]
        # Sould children map being recursively be called after the
        # current node?
        if recursive is Epac.config.recursive_down:
            [child.map(rec_down=True, **kwargs) for child in self.children]
        return data_out


class ParKFold(ParRowSlicer, ParNodeFactory):
    """ KFold parallelization node"""
    train_data_suffix = "train"
    test_data_suffix = "test"

    def __init__(self, n=None, n_folds=None, slices=None, nb=None, **kwargs):
        super(ParKFold, self).__init__(slices=slices,
            name="KFold-" + str(nb), **kwargs)
        self.n = n
        self.n_folds = n_folds

    def produceParNodes(self):
        nodes = []
        from sklearn.cross_validation import KFold  # StratifiedKFold
        nb = 0
        for train, test in KFold(n=self.n, n_folds=self.n_folds):
            nodes.append(ParKFold(slices={ParKFold.train_data_suffix: train,
                                          ParKFold.test_data_suffix: test},
                                          nb=nb))
            nb += 1
        return nodes

    @classmethod
    def split_train_test(cls, **kwargs):
        """Split kwargs into train dict (that contains train suffix in kw)
        and test dict (that contains test suffix in kw).

        Returns
        -------
        Two dictionaries without the train and test suffix into kw. Outputs
        are then compliant with estimator API that takes only X, y paramaters.
        If only "X" an "y" kw are found they are replicated into the both
        outputs.

        Example
        -------
       >>> ParKFold.split_train_test(Xtrain=1, ytrain=2, Xtest=-1, ytest=-2,
       ... a=33)
       ({'y': 2, 'X': 1, 'a': 33}, {'y': -2, 'X': -1, 'a': 33})

        >>> ParKFold.split_train_test(X=1, y=2, a=33)
        ({'y': 2, 'X': 1, 'a': 33}, {'y': 2, 'X': 1, 'a': 33})
        """
        kwargs_train = kwargs.copy()
        kwargs_test = kwargs.copy()
        for data_key in Epac.config.kwargs_data_prefix:
            data_key_train = data_key + ParKFold.train_data_suffix
            data_key_test = data_key + ParKFold.test_data_suffix
            if data_key_test in kwargs_train:  # Remove [X|y]test from train
                kwargs_train.pop(data_key_test)
            if data_key_train in kwargs_test:  # Remove [X|y]train from test
                kwargs_test.pop(data_key_train)
            # [X|y]train becomes [X|y] to be complient with estimator API
            if data_key_train in kwargs_train:
                kwargs_train[data_key] = kwargs_train.pop(data_key_train)
            # [X|y]test becomes [X|y] to be complient with estimator API
            if data_key_test in kwargs_test:
                kwargs_test[data_key] = kwargs_test.pop(data_key_test)
        return kwargs_train, kwargs_test

    @classmethod
    def join_train_test(cls, kwargs_train, kwargs_test):
        """Merge train test separate kwargs into single dict.

            Returns
            -------
            A single dictionary with train and test suffix as kw.

            Example
            -------
            >>> ParKFold.join_train_test(dict(X=1, y=2, a=33),
            ...                          dict(X=-1, y=-2, a=33))
            {'ytest': -2, 'Xtest': -1, 'a': 33, 'Xtrain': 1, 'ytrain': 2}
        """
        kwargs = dict()
        for data_key in Epac.config.kwargs_data_prefix:
            if data_key in kwargs_train:  # Get [X|y] from train
                data_key_train = data_key + ParKFold.train_data_suffix
                kwargs[data_key_train] = kwargs_train.pop(data_key)
            if data_key in kwargs_test:  # Get [X|y] from train
                data_key_test = data_key + ParKFold.test_data_suffix
                kwargs[data_key_test] = kwargs_test.pop(data_key)
        kwargs.update(kwargs_train)
        kwargs.update(kwargs_test)
        return kwargs


class ParStratifiedKFold(ParRowSlicer):
    def __init__(self, slices, nb, **kwargs):
        super(ParStratifiedKFold, self).__init__(slices=slices,
            name="KFold-" + str(nb), **kwargs)


class ParPermutation(ParRowSlicer, ParNodeFactory):
    """ Permutation parallelization node

    2. implement the nodes ie.: the methods
       - fit and transform that modify the data during the "map" phase: the
         top-down (root to leaves) data flow
       - reduce that locally agregates the map results during the "reduce"
         phase: the bottom-up (leaves to root) data-flow.
    """
    def __init__(self, n=None, n_perms=None, permutation=None, nb=None,
                 **kwargs):
        super(ParPermutation, self).__init__(slices=[permutation],
            name="Permutation-" + str(nb), **kwargs)
        self.n = n
        self.n_perms = n_perms

    def produceParNodes(self):
        nodes = []
        from addtosklearn import Permutation
        nb = 0
        for perm in Permutation(n=self.n, n_perms=self.n_perms):
            nodes.append(ParPermutation(permutation=perm, nb=nb))
            nb += 1
        return nodes


def reducefunc(key, val):
    
    val = r['y_pred']
    mean_pred = np.asarray(val['y_pred'])
    mean_true = np.asarray(val['y_true'])
    accuracies = np.sum(mean_true == mean_pred, axis=-1)
    accuracies_cv_mean = np.mean(accuracies, axis=-1)
    accuracies_perm_pval = np.sum(accuracies_cv_mean[1:] >
        accuracies_cv_mean[0])
    return dict(method=key2, accuracies_cv_mean=accuracies_cv_mean,
                accuracies_perm_pval=accuracies_perm_pval)

# Data
X = np.asarray([[1, 2], [3, 4], [5, 6], [7, 8], [-1, -2], [-3, -4], [-5, -6], [-7, -8]])
y = np.asarray([1, 1, 1, 1, -1, -1, -1, -1])

from sklearn import svm
steps = (ParKFold(n=X.shape[0], n_folds=4),
         svm.SVC(kernel='linear'))

tree = Epac(steps=steps, store="/tmp/store")
#tree2 = Epac(store="/tmp/store")
[leaf.map(X=X, y=y) for leaf in tree]
res = tree.reduce()
tree.map(rec_up=False, rec_down=True)
