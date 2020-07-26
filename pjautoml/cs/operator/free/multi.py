import pjml.stream.transform.multi as ml
from pjautoml.cs.operator.free.container import Container


class Multi(Container):
    """TODO."""

    def __init__(self, *args, seed=0, **kwargs):
        super().__init__(
            *args, seed=seed, name=ml.Multi.__name__, path=ml.Multi.__module__, **kwargs
        )
