from pjml.cs.singleton import Singleton
from pjml.abs.asoperand import AsOperand


class AsOperandCS(AsOperand):
    def __add__(self, other):
        from pjautoml.cs.operator.free.select import Select

        # if both are single --> go to ml level
        if isinstance(self.cs, Singleton) and isinstance(other.cs, Singleton):
            return AsOperand.__add__(self, other)
        if isinstance(other, Select):
            return Select(self, *other.css)
        elif isinstance(self, Select):
            return Select(*self.css, other)
        return Select(self, other)

    def __mul__(self, other):
        from pjautoml.cs.operator.free.chain import Chain

        # if both are single --> go to ml level
        if isinstance(self.cs, Singleton) and isinstance(other.cs, Singleton):
            return AsOperand.__mul__(self, other)
        if isinstance(other, Chain):
            return Chain(self, *other.css)
        if isinstance(self, Chain):
            return Chain(*self.css, other)
        return Chain(self, other)

    def __matmul__(self, other):  # @
        from pjautoml.cs.operator.free.shuffle import Shuffle

        # if both are single --> go to ml level
        if isinstance(self.cs, Singleton) and isinstance(other.cs, Singleton):
            return AsOperand.__matmul__(self, other)
        if isinstance(other, Shuffle):
            return Shuffle(self, *other.css)
        elif isinstance(self, Shuffle):
            return Shuffle(*self.css, other)
        return Shuffle(self, other)

    # Ensures resulting object will also accept operators.
    def __radd__(self, other):
        return self.__class__.__add__(other, self)

    def __rmul__(self, other):
        return self.__class__.__mul__(other, self)

    def __rmatmul__(self, other):
        return self.__class__.__matmul__(other, self)
