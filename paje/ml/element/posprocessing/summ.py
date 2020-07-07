import traceback

import numpy

from paje.searchspace.hp import CatHP
from paje.searchspace.configspace import ConfigSpace
from paje.ml.element.element import Element
from paje.util.distributions import choice


class Summ(Element):
    _functions = {
        'mean': numpy.mean
    }

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self._function = self._functions[self.config['function']]
        self.field = self.config['field']

    def apply_impl(self, data):
        return self.use_impl(data)

    def use_impl(self, data):
        chain, values = data.C.pop()
        try:
            return data.updated(
            self,
            C=chain,
            **{self.field: self._function(values)})
        except TypeError as e:
            if str(e).__contains__('cannot perform reduce with flexible type'):
                traceback.print_exc()
                print(values)
                print('W: TODO: put a correct msg here : You probably have to '
                      'reset your storage due to '
                      'incompatible versions of Pajé.')
                exit(0)

    @classmethod
    def cs_impl(cls):
        hps = [
            CatHP('function', choice, itens=['mean'])
        ]
        return ConfigSpace(name=cls.__name__, hps=hps)

    @classmethod
    def isdeterministic(cls):
        return True

    def modifies(self, op):
        return [self.field]
