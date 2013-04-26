# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:54:35 2013

Stores for EPAC

@author: edouard.duchesnay@cea.fr
"""

import os
import glob
import pickle
import json
import inspect
import numpy as np
from abc import abstractmethod


class Store(object):
    """Abstract Store"""

    @abstractmethod
    def save(self, obj, key):
        """Store abstract method"""

    @abstractmethod
    def load(self, key):
        """Store abstract method"""


class StoreLo(Store):
    """ Store based on Living Objects"""

    def save(self, obj, key):
        raise ValueError("Not implemented")

    def load(self, key):
        raise ValueError("Not implemented")


class StoreFs(Store):
    """ Store based of file system"""

    def __init__(self):
        pass

    def key2path(self, key):
        from epac.workflow.base import key_split
        prot, path = key_split(key)
        return path

    def save(self, obj, key, protocol="txt"):
        """ Save object

        Parameters
        ----------

        obj:
            object to be saved

        key: str
            The primary key

        protocol: str
            "txt": try with JSON if fail use "bin": (pickle)
        """
        path = self.key2path(key)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        # JSON
        from epac.workflow.base import conf
        if protocol is "txt":
            file_path = path + conf.STORE_FS_JSON_SUFFIX
            json_failed = self.save_json(obj, file_path)
        if protocol is "bin" or json_failed:
            # saving in json failed => pickle
            file_path = path + conf.STORE_FS_PICKLE_SUFFIX
            self.save_pickle(obj, file_path)

    def load(self, key):
        """Load everything that is prefixed with key.

        Parmaters
        ---------
        key: str
            prefix to find files.

        Return
        ------
            A dictonnary of all loaded objects, where keys are the differences
            between file names prefixed with the "key" parameters. If the "key"
            parameters match exactly a file, the retruned dict will contain a
            single object with an empty key.
        """
        from epac.workflow.base import conf
        path = self.key2path(key)
        #prefix = os.path.join(path, conf.STORE_FS_NODE_PREFIX)
        if os.path.isdir(path):
            path = path + os.path.sep
        # Get all files
        file_paths = [f for f in glob.glob(path + '*')  if os.path.isfile(f)]
        loaded = dict()
        for file_path in file_paths:
            _, ext = os.path.splitext(file_path)
            if ext == conf.STORE_FS_JSON_SUFFIX:
                name = file_path.replace(path, "").\
                    replace(conf.STORE_FS_JSON_SUFFIX, "")
                obj = self.load_json(file_path)
                loaded[name] = obj
            elif ext == conf.STORE_FS_PICKLE_SUFFIX:
                name = file_path.replace(path, "").\
                    replace(conf.STORE_FS_PICKLE_SUFFIX, "")
                loaded[name] = self.load_pickle(file_path)
            else:
                raise IOError('File %s has an unkown extension: %s' %
                    (file_path, ext))
        return loaded

    def save_pickle(self, obj, file_path):
        output = open(file_path, 'wb')
        pickle.dump(obj, output)
        output.close()

    def load_pickle(self, file_path):
        #u'/tmp/store/KFold-0/SVC/__node__NodeEstimator.pkl'
        inputf = open(file_path, 'rb')
        obj = pickle.load(inputf)
        inputf.close()
        return obj

    def save_json(self, obj, file_path):
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


def get_store(key):
    """ factory function returning the Store object of the class
    associated with the key parameter"""
    from epac.workflow.base import  key_split, conf
    splits = key_split(key)
    if len(splits) != 2 and \
        not(splits[0] in (conf.KEY_PROT_FS, conf.KEY_PROT_MEM)):
        raise ValueError('No valid storage has been associated with key: "%s"'
            % key)
    prot, path = splits
    if prot == conf.KEY_PROT_FS:
        return StoreFs()
#    FIXME
#    elif prot == conf.KEY_PROT_MEM:
#        return StoreLo(storage_root=_Node.roots[path])
    else:
        raise ValueError("Invalid value for key: should be:" +\
        "lo for no persistence and storage on living objects or" +\
        "fs and a directory path for file system based storage")


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