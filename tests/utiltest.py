from paje.ml.element.element import Element
from paje.searchspace.configspace import HPTree
import numpy as np


class SimpElem(Element):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.count_apply = 0
        self.count_use = 0
        self.count_buid = 0
        self.count_tree = 0
        self.model = None

    def build_impl(self, **config):
        self.count_buid += 1
        print(">>>>> ", self.config)
        self.oper = self.config['oper']

    def apply_impl(self, data):
        self.count_apply += 1

        value = data.X
        if self.oper == '+':
            value = data.X + data.X
        elif self.oper == '*':
            value = data.X * data.X
        elif self.oper == '/':
            value = data.X / data.X
        elif self.oper == '-':
            value = data.X - data.X
        elif self.oper == '.':
            value = np.dot(data.X, data.X)

        self.model = value.copy()
        data = data.updated(self, X=value)
        return data

    def use_impl(self, data):
        self.count_use += 1
        value = self.model + data.X
        data = data.updated(self, X=value)
        return data

    def cs_impl(self):
        return HPTree({'oper': ['c', ['+', '-', '*', '.']]}, [])


