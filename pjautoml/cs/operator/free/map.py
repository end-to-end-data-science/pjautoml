import pjml.stream.transform.map as ml
from pjautoml.cs.operator.free.container import Container


class Map(Container):
    """TODO."""

    def __init__(self, *args, seed=0, **kwargs):
        super().__init__(*args, seed=seed, name=ml.Map.__name__, path=ml.Map.__module__, **kwargs)
