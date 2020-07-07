from numpy.random import randint


def choice(items):
    idx = randint(0, len(items))
    return items[idx]