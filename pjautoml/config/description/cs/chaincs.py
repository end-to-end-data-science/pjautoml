from pjautoml.config.description.cs.abc.operatorcs import OperatorCS


class ChainCS(OperatorCS):
    """A Chain is sampled."""

    def sample(self):
        components = [cs.sample() for cs in self.components]
        from pjml.tool.chain import Chain
        return Chain(components=components)

