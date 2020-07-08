import traceback
from functools import partial

import numpy

from pjdata.mixin.printing import withPrinting


class Param(withPrinting):
    """Base class for all kinds of algorithm (hyper)parameters."""

    def __init__(self, function, **kwargs):
        self._jsonable = kwargs.copy()
        # TODO: Should we also add the function module ?
        self._jsonable["function"] = function.__name__
        self._jsonable["module"] = function.__module__
        self.function = partial(function, **kwargs)
        self.kwargs = kwargs

    def _jsonable_impl(self):
        return self._jsonable

    def sample(self):
        try:
            return self.function()
        except Exception as e:
            traceback.print_exc()
            print(e)
            print("Problems sampling: ", self)
            exit(0)


class CatP(Param):
    pass


class SubP(Param):
    """Subset of values."""

    pass


class PermP(Param):
    """Permutation of a list."""

    pass


class OrdP(Param):
    pass


class RealP(Param):
    pass


class IntP(Param):
    def sample(self):
        try:
            return int(numpy.round(self.function()))
        except Exception as e:
            traceback.print_exc()
            print(e)
            print("Problems sampling: ", self)
            exit(0)


class FixedP(Param):
    def __init__(self, value):
        # TODO: Should it return an implemented function?
        #  Otherwise, in json formti, function will be  = <lambda>
        super().__init__(lambda: value)
