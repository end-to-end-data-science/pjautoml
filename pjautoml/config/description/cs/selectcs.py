from pjautoml.config.description.cs.abc.operatorcs import OperatorCS


class SelectCS(OperatorCS):
    """Only one CS is sampled."""

    def sample(self):
        from pjautoml.config.description.distributions import choice
        cs = choice(self.components)
        return cs.sample()
