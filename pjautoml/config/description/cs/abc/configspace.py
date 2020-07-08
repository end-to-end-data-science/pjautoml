from abc import abstractmethod
from functools import lru_cache

from pjdata.aux.util import Property
from pjdata.mixin.printing import withPrinting


class ConfigSpace(withPrinting):
    """Tree representing a (probably infinite) set of (hyper)parameter spaces.
    """

    _name = None

    def __init__(self, jsonable):
        jsonable.update(cs=self.__class__.__name__[0:-2].lower())
        self._jsonable = jsonable

    def _jsonable_impl(self):
        return self._jsonable

    @abstractmethod
    def sample(self):
        pass

    @Property
    def cs(self):
        """Shortcut to ease retrieving a CS from a Transformer class without
        having to check that it is not already a CS."""
        return self

    # Amenities.
    @Property
    @lru_cache()
    def name(self):
        if self._name is None:
            self._name = self.__class__.__name__[0:-2].lower()
        return self._name

    @Property
    @lru_cache()
    def longname(self):
        long = ""
        for component in ["components"]:
            if component in self._jsonable_impl:
                items = ", ".join(tr.longname for tr in self._jsonable_impl[component])
                long = f"[{items}]"
        return self.name + long
