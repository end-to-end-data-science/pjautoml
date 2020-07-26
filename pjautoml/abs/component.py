from abc import abstractmethod
from functools import lru_cache

import pjml.abs.component as c
from pjautoml.abs.mixin.asoperand import AsOperandCS
from pjautoml.cs.operand.graph.graph import Graph
from pjdata.aux.decorator import classproperty
from pjml.cs.cs import CS


class Component(c.Component, AsOperandCS):
    """ TODO.
    """

    def __init__(
        self,
        config: dict,
        enhance: bool = True,
        model: bool = True,
        deterministic: bool = False,
        nodata_handler: bool = False,
    ):
        super().__init__(config, enhance, model, deterministic, nodata_handler)

    @classmethod
    @abstractmethod
    def _cs_impl(cls) -> CS:
        """ TODO.
        """

    @classproperty
    @lru_cache()
    def cs(cls) -> Graph:
        """ TODO.
        """
        cs_ = cls._cs_impl()
        # TODO: Why do we send the 'cls' to the CS contructor avoiding to call 'identified' ?
        return cs_.identified(name=cls.__name__, path=cls.__module__)
