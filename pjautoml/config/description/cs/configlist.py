from collections import Iterable

from pjautoml.config.description.cs.abc.configspace import ConfigSpace
from pjautoml.config.description.distributions import choice


class ConfigList(ConfigSpace):
    """Traversable discrete finite CS.

    Iterable CS. This CS does not accept config spaces, only components.

    components
        A list of components.
    """

    def __init__(self, *args, components=None):
        if components is None:
            components = args
        if isinstance(components, Iterable):
            components = [comp for comp in components]
        super().__init__({'components': components})

        from pjml.tool.abs.component import Component
        for component in components:
            if not (isinstance(component, Component)):
                raise Exception(
                    f'\nGiven: {type(component)}\n{component}\n'
                    f'ConfigList does not accept config spaces, '
                    f'only components!')
        self.current_index = -1
        self.size = len(components)
        self.components = components
        self._name = 'list'

    def sample(self):
        return choice(self.components)

    def __iter__(self):  # TODO: Make ConfigList scalable through some sort of generator solution.
        return self.components.__iter__()

    # def __next__(self):
    #     self.current_index += 1
    #     if self.current_index >= self.size:
    #         self.current_index = -1
    #         raise StopIteration('No more objects left.')
    #     return self.components[self.current_index]
    #
    # def __len__(self):
    #     return self.size
    #
    # def __getitem__(self, idx):
    #     return self.components[idx]
