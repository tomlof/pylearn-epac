# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:54:35 2013

Stores for EPAC

@author: edouard.duchesnay@cea.fr
"""

import os
import shutil
import pickle
import json
import inspect
import numpy as np
from abc import abstractmethod

class Store(object):
    """Abstract Store"""

    @abstractmethod
    def save(self, key, obj, merge=False):
        """Store abstract method"""

    @abstractmethod
    def load(self, key):
        """Store abstract method"""


class StoreMem(Store):
    """ Store based on memory"""

    def __init__(self):
        self.dict = dict()

    def save(self, key, obj, merge=False):
        if not merge or not (key in self.dict):
            self.dict[key] = obj
        else:
            v = self.dict[key]
            if isinstance(v, dict):
                v.update(obj)
            elif isinstance(v, list):
                v.append(obj)

    def load(self, key):
        try:
            return self.dict[key]
        except KeyError:
            return None

class StoreFs(Store):
    """ Store based of file system"""

    def __init__(self, dirpath, clear=False):
        """
        dirpath: str
            Root directory within file system
        
        clear: boolean
            If True clear (delete) everything under the root directory.
        """
        self.dirpath = dirpath
        if clear:
            shutil.rmtree(self.dirpath)
        if not os.path.isdir(self.dirpath):
            os.mkdir(self.dirpath)

    def save(self, key, obj, protocol="txt", merge=False):
        """ Save object

        Parameters
        ----------

        key: str
            The primary key

        obj:
            object to be saved

        protocol: str
            "txt": try with JSON if fail use "bin": (pickle)
        """
        #path = self.key2path(key)
        path = os.path.join(self.dirpath, key)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        # JSON
        from epac.configuration import conf
        if protocol is "txt":
            file_path = path + conf.STORE_FS_JSON_SUFFIX
            json_failed = self.save_json(file_path, obj)
        if protocol is "bin" or json_failed:
            # saving in json failed => pickle
            file_path = path + conf.STORE_FS_PICKLE_SUFFIX
            self.save_pickle(file_path, obj)

    def load(self, key=""):
        """Load everything that is prefixed with key.

        Parmaters
        ---------
        key: str
            if key point to a file (without the extension), return the file
            if key point to a directory, return a dictionary where
            values are objects corresponding to all files found in all
            sub-directories. Values are indexed with their keys.
            if key is an empty string, assume dirpath is a tree root.

        See Also
        --------
        BaseNode.save()
        """
        from epac.configuration import conf
        from epac.workflow.base import key_pop
        path = os.path.join(self.dirpath, key)
        #prefix = os.path.join(path, conf.STORE_FS_NODE_PREFIX)
        if os.path.isfile(path + conf.STORE_FS_PICKLE_SUFFIX):
            return self.load_pickle(path + conf.STORE_FS_PICKLE_SUFFIX)
        if os.path.isfile(path + conf.STORE_FS_JSON_SUFFIX):
            return self.load_pickle(path + conf.STORE_FS_JSON_SUFFIX)
        if os.path.isdir(path):
            filepaths = []
            for base, dirs, files in os.walk(self.dirpath):
                #print base, dirs, files
                for filepath in [os.path.join(base, basename) for \
                    basename in files]:
                    filepaths.append(filepath)
            loaded = dict()
            dirpath = os.path.join(self.dirpath, "")
            for filepath in filepaths:
                _, ext = os.path.splitext(filepath)
                if ext == conf.STORE_FS_JSON_SUFFIX:
                    key1 = filepath.replace(dirpath, "").\
                        replace(conf.STORE_FS_JSON_SUFFIX, "")
                    obj = self.load_json(filepath)
                    loaded[key1] = obj
                elif ext == conf.STORE_FS_PICKLE_SUFFIX:
                    key1 = filepath.replace(dirpath, "").\
                        replace(conf.STORE_FS_PICKLE_SUFFIX, "")
                    loaded[key1] = self.load_pickle(filepath)
                else:
                    raise IOError('File %s has an unkown extension: %s' %
                        (filepath, ext))
            if key == "":  # No key provided assume a whole tree to load
                tree = loaded.pop(conf.STORE_EXECUTION_TREE_PREFIX)
                for key1 in loaded:
                    key, attrname = key_pop(key1)
                    #attrname, ext = os.path.splitext(basename)
                    if attrname != conf.STORE_STORE_PREFIX:
                        raise ValueError('Do not know what to do with %s') \
                            % key1
                    node = tree.get_node(key)
                    if not node.store:
                        node.store = loaded[key1]
                    else:
                        keys_local = node.store.dict.keys()
                        keys_disk = loaded[key1].dict.keys()
                        if set(keys_local).intersection(set(keys_disk)):
                            raise KeyError("Merge store with same keys")
                        node.store.dict.update(loaded[key1].dict)
                loaded = tree
            return loaded


    def save_pickle(self, file_path, obj):
        output = open(file_path, 'wb')
        pickle.dump(obj, output)
        output.close()

    def load_pickle(self, file_path):
        #u'/tmp/store/KFold-0/SVC/__node__NodeEstimator.pkl'
        inputf = open(file_path, 'rb')
        obj = pickle.load(inputf)
        inputf.close()
        return obj

    def save_json(self, file_path,  obj):
        obj_dict = obj_to_dict(obj)
        output = open(file_path, 'wb')
        try:
            json.dump(obj_dict, output)
        except TypeError:  # save in pickle
            output.close()
            os.remove(file_path)
            return 1
        output.close()
        return 0

    def load_json(self, file_path):
        inputf = open(file_path, 'rb')
        obj_dict = json.load(inputf)
        inputf.close()
        return dict_to_obj(obj_dict)

## ============================== ##
## == Conversion Object / dict == ##
## ============================== ##

# Convert object to dict and dict to object for Json Persistance
def obj_to_dict(obj):
    # Composite objects (object, dict, list): recursive call
    if hasattr(obj, "__dict__") and hasattr(obj, "__class__")\
        and hasattr(obj, "__module__") and not inspect.isfunction(obj): # object: rec call
        obj_dict = {k: obj_to_dict(obj.__dict__[k]) for k in obj.__dict__}
        obj_dict["__class_name__"] = obj.__class__.__name__
        obj_dict["__class_module__"] = obj.__module__
        return obj_dict
    elif inspect.isfunction(obj):                     # function
        obj_dict = {"__func_name__": obj.func_name,
                    "__class_module__": obj.__module__}
        return obj_dict
    elif isinstance(obj, dict):                       # dict: rec call
        return {k: obj_to_dict(obj[k]) for k in obj}
    elif isinstance(obj, (list, tuple)):              # list: rec call
        return [obj_to_dict(item) for item in obj]
    elif isinstance(obj, np.ndarray):                 # array: to list
        return {"__array__": obj.tolist()}
    else:
        return obj


def dict_to_obj(obj_dict):
    if isinstance(obj_dict, dict) and '__class_name__' in obj_dict:  # object
        cls_name = obj_dict.pop('__class_name__')               # : rec call
        cls_module = obj_dict.pop('__class_module__')
        obj_dict = {k: dict_to_obj(obj_dict[k]) for k in obj_dict}
        mod = __import__(cls_module, fromlist=[cls_name])
        obj = object.__new__(eval("mod." + cls_name))
        obj.__dict__.update(obj_dict)
        return obj
    if isinstance(obj_dict, dict) and '__func_name__' in obj_dict:  # function
        func_name = obj_dict.pop('__func_name__')
        func_module = obj_dict.pop('__class_module__')
        mod = __import__(func_module, fromlist=[func_name])
        func = eval("mod." + func_name)
        return func
    if isinstance(obj_dict, dict) and '__array__' in obj_dict:
        return np.asarray(obj_dict.pop('__array__'))
    elif isinstance(obj_dict, dict):                         # dict: rec call
        return {k: dict_to_obj(obj_dict[k]) for k in obj_dict}
    elif isinstance(obj_dict, (list, tuple)):                # list: rec call
        return [dict_to_obj(item) for item in obj_dict]
#    elif isinstance(obj, np.ndarray):                       # array: to list
#        return obj.tolist()
    else:
        return obj_dict