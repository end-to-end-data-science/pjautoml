""" Scaler Module
"""
from abc import ABC

from paje.ml.element.element import Element


class Scaler(Element, ABC):
    def apply_impl(self, data):
        # self.model will be set in the child class
        self.model.fit(*data.Xy)
        return self.use_impl(data)

    def use_impl(self, data):
        return data.updated(self, X=self.model.transform(data.X))

    @classmethod
    def isdeterministic(cls):
        return True

    def modifies(self, op):
        return ['X']
