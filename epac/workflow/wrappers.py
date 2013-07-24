# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 14:43:15 2013

@author: jinpeng
"""

from epac.workflow.base import BaseNode
from epac.utils import _func_get_args_names, train_test_merge, train_test_split
from epac.utils import _sub_dict
from epac.configuration import conf
from epac.map_reduce.results import ResultSet
from epac.workflow.base import key_push


## ================================= ##
## == Wrapper node == ##
## ================================= ##
class Wrapper(BaseNode):
    """Node that wrap any class

    Example
    -------
    >>> from epac.workflow.wrappers import Wrapper
    >>> class Node2Wrap:
    ...     def __init__(self):
    ...         self.a = 1
    ...         self.b = 2
    ...     def transform(self, X, y):
    ...         return dict(res = x * self.a + y * self.b)
    ...
    >>> wrapper_node = Wrapper(Node2Wrap())
    >>> print wrapper_node.get_signature()
    Node2Wrap
    >>> print wrapper_node.get_parameters()
    {'a': 1, 'b': 2}

    """

    def __init__(self, wrapped_node):
        super(Wrapper, self).__init__()
        self.wrapped_node = wrapped_node

    def get_signature(self):
        """Overload the base name method"""
        if not self.signature_args:
            return self.wrapped_node.__class__.__name__
        else:
            args_str = ",".join([str(k) + "=" + str(self.signature_args[k])
                             for k in self.signature_args])
            args_str = "(" + args_str + ")"
            return self.wrapped_node.__class__.__name__ + args_str

    def get_parameters(self):
        return self.wrapped_node.__dict__


class TransformNode(Wrapper):
    '''Wrapping a class has a "transform" method

    Example
    -------

    >>> from epac.workflow.wrappers import TransformNode
    >>> from epac.utils import _func_get_args_names
    >>> class Node2Wrap:
    ...     def __init__(self):
    ...         self.a = 1
    ...         self.b = 2
    ...     def transform(self, x, y):
    ...         return dict(res = x * self.a + y * self.b)
    ...
    >>> node = TransformNode(Node2Wrap())
    >>> node.transform(x=1, y=2)
    {'y': 2, 'x': 1, 'res': 5}
    >>> from epac.configuration import conf
    >>> from epac.utils import train_test_merge
    >>> train_test_merged = train_test_merge(dict(x=1, y=2), dict(x=33, y=44))
    >>> print train_test_merged
    {'x/test': 33, 'x/train': 1, 'y/train': 2, 'y/test': 44}
    >>> train_test_merged[conf.KW_SPLIT_TRAIN_TEST] = True
    >>> node.transform(**train_test_merged)
    {'x/test': 33, 'x/train': 1, 'y/train': 2, 'res/train': 5, 'split_train_test': True, 'y/test': 44, 'res/test': 121}

    '''

    def __init__(self, wrapped_node, in_args_transform=None):
        '''
        Parameters
        ----------
        in_args_transform: list of strings
            names of input arguments of the tranform method. If missing,
            discover it automatically.
        '''
        if not hasattr(wrapped_node, "transform"):
            raise ValueError("wrapped_node should implement transform")
        super(TransformNode, self).__init__(wrapped_node=wrapped_node)
        self.in_args_transform = \
            _func_get_args_names(self.wrapped_node.transform) \
            if in_args_transform is None else in_args_transform

    def transform(self, **Xy):
        """
        Parameter
        ---------
        Xy: dictionary
            parameters for transform
        """
        if conf.KW_SPLIT_TRAIN_TEST in Xy:
            Xy_train, Xy_test = train_test_split(Xy)
            # catch args_transform in ds, transform, store output in a dict
            Xy_out_tr = self.wrapped_node.transform(**_sub_dict(Xy_train,
                            self.in_args_transform))
            Xy_out_te = self.wrapped_node.transform(**_sub_dict(Xy_test,
                            self.in_args_transform))
            if type(Xy_out_tr) is not dict or type(Xy_out_te) is not dict:
                raise ValueError("%s.transform should return a dictionary"
                % (self.wrapped_node.__class__.__name__))
            Xy_out = train_test_merge(Xy_out_tr, Xy_out_te)
        else:
            # catch args_transform in ds, transform, store output in a dict
            Xy_out = self.wrapped_node.transform(**_sub_dict(Xy,
                                                 self.in_args_transform))
            if type(Xy_out) is not dict:
                raise ValueError("%s.transform should return a dictionary"
                % (self.wrapped_node.__class__.__name__))
        # update ds with transformed values
        Xy.update(Xy_out)
        return Xy


if __name__ == "__main__":
    import doctest
    doctest.testmod()
