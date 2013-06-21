#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on 20 June 2013

@author: jinpeng.li@cea.fr
@author: edouard.duchesnay@cea.fr
@author: benoit.da_mota@inria.fr

"""
import numpy as np


def _is_numeric(obj):
    return isinstance(obj, (int, long, float, complex))


def _is_dict_or_array_or_list(obj):
    if type(obj) is np.ndarray:
        return True
    if type(obj) is list:
        return True
    if type(obj) is dict:
        return True
    return False


def _is_array_or_list(obj):
    if type(obj) is np.ndarray:
        return True
    if type(obj) is list:
        return True
    return False


def isequal(obj1, obj2):
    _EPSILON = 0.00001
    if _is_numeric(obj1):
        if (np.absolute(obj1 - obj2) > _EPSILON):
            return False
        else:
            return True
    elif (isinstance(obj1, dict)):
        for key in obj1.keys():
            if not isequal(obj1[key], obj2[key]):
                return False
        return True
    elif (_is_array_or_list(obj1)):
        obj1 = np.asarray(list(obj1))
        obj2 = np.asarray(list(obj2))
        for index in xrange(len(obj1.flat)):
            if not isequal(obj1.flat[index], obj2.flat[index]):
                return False
        return True
    else:
        return obj1 == obj2


def comp_2wf_reduce_res(wf1, wf2):
    res_wf1 = wf1.reduce()
    res_wf2 = wf2.reduce()
    return isequal(res_wf1, res_wf2)


def displayres(d, indent=0):
    print repr(d)
#    for key, value in d.iteritems():
#        print '\t' * indent + str(key)
#        if isinstance(value, dict):
#            displayres(value, indent + 1)
#        else:
#            print '\t' * (indent + 1) + str(value)