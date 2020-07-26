from pjdata.mixin.printing import withPrinting
from pjml.util.distributions import choice


class Node(withPrinting):
    """Partial settings for a component.

    Parameters
    ----------
    params
        Dictionary like {'param1': Param(...), 'param2': Param(...), ...}.
    children
        List of the next nodes. Only one is sampled at a time.
    """

    def __init__(self, params=None, children=None):
        self._jsonable = params.copy()
        self._jsonable['children'] = children
        self.params = {} if params is None else params
        self.children = [] if children is None else children
        if any([not isinstance(cs, Node) for cs in self.children]):
            raise Exception('Node can only have Nodes as children.')

    def _jsonable_impl(self):
        return self._jsonable

    def partial_sample(self):
        """Sample a partial config. It is not enough to make a component.

        Returns
        -------
        A dict containing the partial config.
        """
        config = {}

        # Fill config with values from child nodes.
        if self.children:
            child_node = choice(self.children)
            config.update(child_node.partial_sample())

        # Complete args with current node values, possibly overriding some
        # values from children nodes (this happens with frozen cs()).
        for name, param in self.params.items():
            config[name] = param.sample()

        return config

    def updated(self, **kwargs):
        dic = {
            'params': self.params,
            'children': self.children
        }
        dic.update(kwargs)
        return self.__class__(**dic)
