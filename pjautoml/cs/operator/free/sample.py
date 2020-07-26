from pjautoml.cs.operand.list.flist import FList


class Sample(FList):
    """TODO."""

    def __init__(self, cs, n=100):
        css = tuple(cs.cs.sample() for _ in range(n))
        super().__init__(*css)
        self.cs = self.cs.cs
        self.n = n
