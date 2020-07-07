""" Component module.
"""
import os
from abc import ABC, abstractmethod

from numpy.random import choice

from paje.searchspace.hp import CatHP, FixedHP
from paje.util.encoders import uuid, json_pack


class Component(ABC):
    """Todo the docs string
    """

    def __init__(self, config):
        self.config = config.copy()

        # 'mark' should not identify components, it only marks results.
        # 'mark', 'max_time', 'class', 'module' are reserved word
        self.mark = self.config.pop('mark') if 'mark' in config else None
        self.max_time = self.config.pop('max_time') \
            if 'max_time' in config else None

        # 'random_state' is reserved word
        if self.isdeterministic() and 'random_state' in self.config:
            del self.config['random_state']

        if 'class' not in config:
            self.config['class'] = self.__class__.__name__
        if 'module' not in config:
            self.config['module'] = self.__module__

        self._serialized = json_pack(self.config)
        self.uuid = uuid(self._serialized.encode())

        self.name = self.config['class']
        self.module = self.config['module']

        self._modified = {'a': None, 'u': None}

        # self.model here refers to classifiers, preprocessors and, possibly,
        # some representation of pipelines or the autoML itself.
        # Another possibility is to generalize modules to a new class Module()
        # that has self.model.
        self.unfit = True
        self.model = None

        # Each apply() uses a different training data, so this uuid is mutable
        self._train_data_uuid__mutable = None

        self.locked_by_others = False
        self.failed = False
        self.time_spent = None
        self.host = None
        self.failure = None
        self._param = None

    @classmethod
    def cs(cls, **kwargs):
        """
        Each tree represents a set of hyperparameter spaces and is a, possibly
        infinite, set of configurations.
        Parameters
        ----------
        data
            If given, 'data' limits the search interval of some hyperparameters

        Returns
        -------
            Tree representing all the possible hyperparameter spaces.
        """
        tree = cls.cs_impl(**kwargs)
        if 'config_spaces' in kwargs:
            del kwargs['config_spaces']
        hps = tree.hps.copy()

        hps['module'] = CatHP(choice, a=[cls.__module__])
        hps['class'] = CatHP(choice, a=[cls.__name__])

        # Freeze args passed via kwargs
        for k, v in kwargs.items():
            hps[k] = FixedHP(v)

        return tree.updated(hps=hps)

    def param(self):
        if self._param is None:
            self._param = self.config.copy()
            del self._param['class']
            del self._param['module']

        return self._param

    @classmethod
    def cfg(cls, **kwargs):
        kwargs['class'] = cls.__name__
        kwargs['module'] = cls.__module__
        return kwargs

    @classmethod
    def isdeterministic(cls):
        return False

    @classmethod
    @abstractmethod
    def cs_impl(cls, **kwargs):
        """Todo the doc string
        """
        pass

    def __str__(self, depth=''):
        return self.name + " " + str(self.config)

    __repr__ = __str__

    def serialized(self):
        return self._serialized

    @staticmethod
    def clock():
        t = os.times()
        # return t[4]  # Wall time
        return t[0] + t[1] + t[2] + t[3]

    def train_data_uuid__mutable(self):
        if self._train_data_uuid__mutable is None:
            raise Exception('This component should be applied to have '
                            'an internal training data uuid.', self.name)
        return self._train_data_uuid__mutable

    @abstractmethod
    def modifies(self, op):
        pass
