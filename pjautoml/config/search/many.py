"""Operations over many CSs."""
from pjautoml.config.description.cs.selectcs import SelectCS
from pjautoml.config.description.cs.shufflecs import ShuffleCS


def select(*components):
    return SelectCS(*components)


def shuffle(*components):
    return ShuffleCS(*components)
