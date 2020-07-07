from abc import abstractmethod

from paje.base.component import Component


class Iterable(Component):
    @abstractmethod
    def iterations(self, data):
        pass
