# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 16:13:26 2013

@author: edouard.duchesnay@cea.fr
"""
import numpy as np

def range_log2(n, add_n=True):
    """Return log2 range starting from 1"""
    rang = (2**np.arange(int(np.floor(np.log2(n)))+1)).tolist()
    if add_n:
        rang.append(int(n))
    return rang

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

class ListWithMetaInfo(list):
    pass
    
def _list_of_dicts_2_dict_of_lists(list_of_dict, axis_name=None,
                                   axis_values=[]):
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
            result = d[key2]
            # result is a dictionary
            if isinstance(result, dict):
                if not key2 in dict_of_list.keys():
                    dict_of_list[key2] = dict()
                for key3 in result.keys():
                    if not key3 in dict_of_list[key2].keys():
                        dict_of_list[key2][key3] = ListWithMetaInfo()
                        dict_of_list[key2][key3].axis_name = axis_name
                        dict_of_list[key2][key3].axis_values = axis_values
                    dict_of_list[key2][key3].append(result[key3])
            else:  # simply concatenate
                if not key2 in dict_of_list.keys():
                    dict_of_list[key2] = ListWithMetaInfo()
                    dict_of_list[key2].axis_name = axis_name
                    dict_of_list[key2].axis_values = axis_values
                dict_of_list[key2].append(result)
    return dict_of_list


def dict_diff(*dicts):
    """Find the differences in a dictionaries

    Returns
    -------
    diff_keys: a list of keys that differ amongs dicts
    diff_vals: a dict with keys values differences between dictonaries.
        If some dict differ bay keys (some keys are missing), return
        the key associated with None value

    Examples
    --------
    >>> dict_diff(dict(a=1, b=2, c=3), dict(b=0, c=3))
    {'a': None, 'b': [0, 2]}
    >>> dict_diff(dict(a=1, b=[1, 2]), dict(a=1, b=[1, 3]))
    {'b': [[1, 2], [1, 3]]}
    >>> dict_diff(dict(a=1, b=np.array([1, 2])), dict(a=1, b=np.array([1, 3])))
    {'b': [array([1, 2]), array([1, 3])]}
    """
    # Find diff in keys
    union_keys, inter_keys, diff_keys = _list_union_inter_diff(*[d.keys()
                                            for d in dicts])
    diff_vals = dict()
    for k in diff_keys:
        diff_vals[k] = None
    # Find diff in shared keys
    for k in inter_keys:
        if isinstance(dicts[0][k], (np.ndarray, list, tuple)):
            if not np.all([np.all(d[k] == dicts[0][k]) for d in dicts]):
                diff_vals[k] = [d[k] for d in dicts]
        elif isinstance(dicts[0][k], dict):
            if not np.all([d[k] == dicts[0][k] for d in dicts]):
                diff_vals[k] = [d[k] for d in dicts]
        else:
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


def _dict_prefix_keys(prefix, d):
    return {prefix + str(k): d[k] for k in d}


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