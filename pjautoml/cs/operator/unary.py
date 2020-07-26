from abc import ABC

from pjautoml.cs.css import CSS


class Unary(CSS, ABC):
    """TODO."""

    def __init__(self, *css, **kwargs):
        if len(css) != 1:
            raise Exception("You must give only one CS.")

        super().__init__(*css)

        self.kwargs = kwargs
