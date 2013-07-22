# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 16:58:40 2013

@author: jinpeng.li@cea.fr

Factory class to build BaseNode
"""

from epac.workflow.base import BaseNode
from epac.utils import _func_get_args_names
from epac.configuration import conf
from epac.utils import _sub_dict, _as_dict
from epac.utils import _func_get_args_names, train_test_merge, train_test_split, _dict_suffix_keys


## ================================= ##
## == Wrapper node == ##
## ================================= ##
class Wrapper(BaseNode):
    """Node that wrap any class

    Example
    -------
    >>> from epac.workflow.factory import Wrapper
    >>> class Node2Wrap:
    ...     def __init__(self):
    ...         self.a = 1
    ...         self.b = 2
    ...
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

    >>> from epac.workflow.factory import NodeFactory
    >>> from epac.workflow.factory import TransformNode
    >>> class Node2Wrap:
    >>> def __init__(self):
    ...     self.a = 1
    ...     self.b = 2
    >>> def transform(self, x, y):
    ...     # raise error if only return value
    ...     # e.g. return x * self.a + y * self.b
    ...     return dict(res = x * self.a + y * self.b)
    >>> node = TransformNode(Node2Wrap())
    >>> node.transform(x=1, y=2)
    {'y': 2, 'x': 1, 'res': 5}

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
        super(TransformNode, self).__init__(wrapped_node)
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


class NodeFactory:

    @staticmethod
    def build(node):
        if hasattr(node, "transform"):
            return TransformNode(node)



if __name__ == "__main__":
    import doctest
    doctest.testmod()
