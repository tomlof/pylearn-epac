"""
epac : Embarrassingly Parallel Array Computing
"""
print __doc__


import numpy as np

from addtosklearn import Permutation
from sklearn.cross_validation import KFold, StratifiedKFold


class Store(object):
    """ Store based of file system"""
    def __init__(self):
        pass

    def save_map_output(key1, key2=None, val2=None, keyvals2=None):
        """ Collect and store map output.
        Map is called given primary (1) key/val and produce intermediary
        (secondary, 2) key/val. This method intermediary key/val
        indexed by primary key.

        Parameters
        ----------
        key1 : (typically a string) primary key

        key2 : (typically a string) the intermediary key

        val2 : (dictionary, list, tuple or array) the intermediary value
        produced by the mapper.
                If key/val are provided a single map output is added

        keyvals2 : a dictionary of intermediary keys/values produced by the
        mapper.
        """
        if key2 and val2:
            #self.map_outputs[key] = val
            pass
        if keyvals2:
            #self.map_outputs.update(keyvals)
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
        key_prefix, key_content = key.split(":", 1)
        import os
        if not os.path.exists(key_content):
            os.makedirs(key_content)
        return key_content

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

    def save_node(self, node, recursive=True):
        print node.get_key()
        path = self.key2path(node.get_key())
        import os
        class_name = str(node.__class__).split(".")[-1].\
            replace(r"'", "").replace(r">", "")
        filename = Epac.config.store_fs_node_prefix + class_name +\
            Epac.config.store_fs_json_suffix
        file_path = os.path.join(path, filename)
        self.save_json(node.todict(), file_path)
        if recursive and len(node.children):
            for child in node.children:
                self.save_node(child, recursive=True)

    def load_node(self, key, recursive=True):
        """Load a node given a key, recursive=True recursively walk through
        children"""
        path = self.key2path(key)
        import os
        prefix = os.path.join(path, Epac.config.store_fs_node_prefix)
        import glob
        node_path_json = glob.glob(prefix + '*' + \
            Epac.config.store_fs_json_suffix)
        if len(node_path_json) != 1:
            raise ValueError("Found more that one or none node file in " +\
            prefix)
        node_path_json = node_path_json[0]
        file_path = node_path_json
        node_dict = self.load_json(file_path)
        class_str = node_path_json.replace(prefix, "").\
            replace(Epac.config.store_fs_json_suffix, "")
        node = object.__new__(eval(class_str))
        node.__dict__.update(node_dict)
        # If Children: Recursively walk through children
        if recursive and len(node.children):
            for i in xrange(len(node.children)):
                child = node.children[i]
                child_key = Epac.config.key_path_sep.join(
                    [key, child])
                node.children[i] = self.load_node(key=child_key,
                    recursive=True)
        # If leaf node: look for map outputs
        if len(node.children) == 0:
            map_paths = glob.glob(os.path.join(path,
                Epac.config.store_fs_map_output_prefix) + '*')
            for map_path in map_paths:
                ext = os.path.splitext(map_path)[-1]
                if ext == Epac.config.store_fs_pickle_suffix:
                    map_obj = self.load_pickle(map_path)
                if ext == Epac.config.store_fs_json_suffix:
                    map_obj = self.load_json(map_path)
                key2 = os.path.splitext(os.path.basename(map_path))[0].\
                    replace(Epac.config.store_fs_map_output_prefix, "", 1)
                node.map_outputs[key2] = map_obj
        return node

    def save_pickle(self, obj, file_path):
            import pickle
            output = open(file_path, 'wb')
            pickle.dump(obj, output)
            output.close()

    def load_pickle(self, file_path):
            import pickle
            inputf = open(file_path, 'rb')
            obj = pickle.load(inputf)
            inputf.close()
            return obj

    def save_json(self, obj, file_path):
            import json
            output = open(file_path, 'wb')
            json.dump(obj, output)
            output.close()

    def load_json(self, file_path):
            import json
            inputf = open(file_path, 'rb')
            obj = json.load(inputf)
            inputf.close()
            return obj


class EpacFmwk(object):

    def __init__(self, scheme=None, data=None, store=None):
        # No store is provided, assume that storage will be done directly
        # on "living objects", ie.: key prefix "lo:"
        pass


def get_store(key):
    """ factory function returning the Store object of the class
    associated with the key parameter"""
    key_prefix, key_content = key.split(":", 1)
    if key_prefix == "fs":
        return StoreFs()
    elif key_prefix == "lo":
        return StoreLo(storage_root=EpacFmwk.roots[key_content])
    else:
        raise ValueError("Invalid value for key: should be:" +\
        "lo for no persistence and storage on living objects or" +\
        "fs and a directory path for file system based storage")


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
        key_prot_lo = "lo"  # key storage protocol: living object
        key_prot_fs = "fs"  # key storage protocol: file system
        key_path_sep = "/"
        key_prot_path_sep = ":"  # key storage protocol / path separator

    def __init__(self, scheme=None, data=None, store=None, **kwargs):
        self.__dict__.update(kwargs)
        self.parent = None
        self.children = list()
        self.map_outputs = dict()
        # If a scheme is provided: initial construction of the execution tree
        if scheme:
            if not store:
                import string
                import random
                self.name = Epac.config.key_prot_lo +\
                             Epac.config.key_prot_path_sep +\
                    "".join(random.choice(string.ascii_uppercase +
                        string.digits)
                    for x in range(5))
                self.add_children(self.build_execution_tree(scheme, data))
            # store is a string and a valid directory , assume that storage
            # will be done on the file system, ie.: key prefix "fs://"
            elif isinstance(store, str):
                self.name = Epac.config.key_prot_fs +\
                            Epac.config.key_prot_path_sep + store
                self.add_children(self.build_execution_tree(scheme, data))
                self.save_node()
            else:
                raise ValueError("Invalid value for store: should be: " +\
                "None for no persistence and storage on living objects or " +\
                "a string path for file system based storage")

    # Tree operations
    # ---------------
    def add_child(self, child):
        self.children.append(child)

    def add_children(self, children):
        for child in children:
            self.children.append(child)
            child.parent = self

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

    def transform(self, data, compose_from_root=True):
        if compose_from_root and self.parent:  # compose tranfo up to root
            data = self.parent.transform(data, compose_from_root=True)
        # identity transformation
        return data

    def get_name(self):
        return self.name

    def get_key(self):
        if not self.parent:
            return self.get_name()
        return Epac.config.key_path_sep.join(
            [self.parent.get_key(), self.get_name()])

    def get_leaves(self):
        if not len(self.children):
            return [self]
        else:
            leaves = []
            for child in self.children:
                leaves = leaves + child.get_leaves()
            return leaves

    def todict(self):
        ret = self.__dict__.copy()
        ret["children"] = [child.name for child in ret["children"]]
        if self.parent:
            ret["parent"] = self.parent.name
        return ret

    # Tree construction
    def build_execution_tree(self, scheme, data):
        """ Chunk function,
        A scheme is a list of slicers, provided with their parameters
        ((Slicers1, (parameters), (hyper-parameters),
         (Slicers2, (parameters), (hyper-parameters), ...)

        Slicers is a class that provide an iterable object (slicers), each item
        """
        if len(scheme) == 0:
            return []
        nodes = []
        scheme_curr = scheme[0]
        scheme_next = scheme[1:]
        slicers_cls = scheme_curr[0]
        if len(scheme_curr) < 2 or not isinstance(scheme_curr[1], dict):
            raise ValueError("Arg should provided and shoud be a dictionnary")
        slicer_args = scheme_curr[1]
        if len(scheme_curr) < 3:
            slicer_opt_args = dict()
        elif not isinstance(scheme_curr[2], dict):
            raise ValueError("Opt arg should be a dictionnary")
        else:
            slicer_opt_args = scheme_curr[2]
        # Get slicer arguments, eventually eval them
        slicer_args_eval = dict()
        for k in slicer_args.keys():
            if isinstance(slicer_args[k], str):
                slicer_args_eval[k] = eval(slicer_args[k], data.copy())
            else:
                slicer_args_eval[k] = slicer_args[k]
        slicers = object.__new__(slicers_cls)
        slicers.__init__(**slicer_args_eval)
        nb = 0
        for slicer in slicers:
            #slicer = slicers.__iter__().next()
            slicer_node_cls_str = "".join([slicers.__class__.__name__ +
                                           "ParNode"])
            #print slicer_node_cls_str
            slicer_node_cls = globals()[slicer_node_cls_str]
            slicer_node = object.__new__(slicer_node_cls)
            slicer_node.__init__(slicer, nb=nb, **slicer_opt_args)
            data_tr = slicer_node.transform(data, compose_from_root=False)
            slicer_node.add_children(
                self.build_execution_tree(scheme_next, data_tr))
            nodes.append(slicer_node)
            nb += 1
        return nodes

    # Iterate over leaves
    def __iter__(self):
        for leaf in self.get_leaves():
            yield leaf

    # Aggregation operations
    # ----------------------
    def aggregate(self):
        # Terminaison (leaf) node
        if len(self.children) == 0:
            return self.map_outputs
        # 1) Build sub-aggregates over children
        sub_aggregates = [child.aggregate() for child in self.children]
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

    # I/O (persistance) operation
    def save_node(self):
        store = get_store(self.get_key())
        store.save_node(self)


class SlicerParNode(Epac):
    """Parallelization is based on several reslicing of the same dataset:
    Slices can be split (shards) or a resampling of the original datasets.

    Parameters
    transform_only:
    """
    def __init__(self, transform_only=None, **kwargs):
        super(SlicerParNode, self).__init__(**kwargs)
        self.transform_only = transform_only


class RowSlicerParNode(SlicerParNode):
    """Parallelization is based on several row-wise reslicing of the same
    dataset"""

    def __init__(self, slices, **kwargs):
        super(RowSlicerParNode, self).__init__(**kwargs)
        # convert a as list if required
        self.slices =\
            [s.tolist() for s in slices if isinstance(s, np.ndarray)]

    def transform(self, data, compose_from_root=True):
        # Recusively compose the tranformations up to root's tree
        if compose_from_root and self.parent:
            data = self.parent.transform(data, compose_from_root=True)
        # If data is an array: reslices it
        if isinstance(data, np.ndarray):
            if len(self.slices) == 1:
                return data[self.slices[0]]
            else:
                return [data[slice] for slice in self.slices]
        # If data is a dict or a list call transform on each item
        if isinstance(data, dict):
            res = data.copy()
            # if it is a dict and transform_only is not empty restrict
            # transformation to self.transform_only
            if self.transform_only:
                keys = self.transform_only
            else:
                keys = data.keys()
        elif isinstance(data, list):
            res = [None] * len(data)
            keys = range(len(data))
        for k in keys:
            res[k] = self.transform(data[k], compose_from_root=False)
        return res


class KFoldParNode(RowSlicerParNode):
    def __init__(self, slices, nb, **kargs):
        super(KFoldParNode, self).__init__(slices=slices,
            name="KFold-" + str(nb), **kargs)


class StratifiedKFoldParNode(RowSlicerParNode):
    def __init__(self, slices, nb, **kargs):
        super(StratifiedKFoldParNode, self).__init__(slices=slices,
            name="KFold-" + str(nb), **kargs)


class PermutationParNode(RowSlicerParNode):
    def __init__(self, permutation, nb, **kargs):
        super(PermutationParNode, self).__init__(slices=[permutation],
            name="Permutation-" + str(nb), **kargs)
        #self.permutation = permutation


# map and reduce functions
def mapfunc(key1, val1):
    X_test = val1["X"][1]
    y_test = val1["y"][1]
    keyvals2 = dict(
        mean=dict(pred=np.sign(np.mean(X_test, axis=1)), true=y_test),
        med=dict(pred=np.sign(np.median(X_test, axis=1)), true=y_test))
    save_map_output(key1, keyvals2=keyvals2)


def reducefunc(key2, val2):
    mean_pred = np.asarray(val2['pred'])
    mean_true = np.asarray(val2['true'])
    accuracies = np.sum(mean_true == mean_pred, axis=-1)
    accuracies_cv_mean = np.mean(accuracies, axis=-1)
    accuracies_perm_pval = np.sum(accuracies_cv_mean[1:] >
        accuracies_cv_mean[0])
    return dict(method=key2, accuracies_cv_mean=accuracies_cv_mean,
                accuracies_perm_pval=accuracies_perm_pval)

# Data
X = np.asarray([[1, 2], [3, 4], [5, 6], [7, 8], [-1, -2], [-3, -4], [-5, -6], [-7, -8]])
y=np.asarray([1, 1, 1, 1, -1, -1, -1, -1])
data = dict(X=X, y=y)

# Resampling scheme
scheme = ((Permutation, dict(n=X.shape[0], n_perms=10),
                        dict(transform_only=["X"])),
          (StratifiedKFold, dict(y="y", n_folds=2)))


epac = Epac(scheme=scheme, data=data, store="/tmp/store")

# 1) Call mapfunc keyvals2
# ------------------------
# Simply call map func
#[mapfunc(key1=fold.get_key(),
#   val1=fold.transform(data)) for fold in epac.get_leaves()]
[mapfunc(key1=fold.get_key(), val1=fold.transform(data)) for fold in epac]

# or use job lib
from sklearn.externals.joblib import Parallel, delayed
p = Parallel(n_jobs=4)(delayed(mapfunc)(key1=fold.get_key(),
                       val1=fold.transform(data)) for fold in
                       epac)

# 1) Re-load results
# ------------------
store = get_store(epac.get_key())
root2 = store.load_node(epac.get_key())

# 2) Aggregate
keyval2 = root2.aggregate()

## 3) Reduce
[reducefunc(key2, keyval2[key2]) for key2 in keyval2.keys()]