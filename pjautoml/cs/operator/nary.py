from abc import ABC

from pjautoml.cs.css import CSS


class NAry(CSS, ABC):
    """TODO."""

    def __init__(self, *css, **kwargs):
        if len(css) == 0:
            raise Exception("You must give at least one CS.")

        super().__init__(*css)

        self.kwargs = kwargs
