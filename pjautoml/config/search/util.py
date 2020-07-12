from collections import Iterable
from heapq import nsmallest, nlargest
from math import ceil, floor

from pjdata.aux.util import _
from pjdata.content.specialdata import NoData
from pjml.config.description.cs.configlist import ConfigList


def compare(x, y):
    if isinstance(x, Iterable) and isinstance(y, Iterable):
        return all(compare(i, j) for i, j in zip(x, y))
    # return x.isequal(y)
    return x == y


def run(clist, train=NoData, test=NoData):
    if not isinstance(clist, ConfigList):
        raise Exception("Exhaustive search is only possible on FiniteCS!")

    results = []
    for pipe in clist:
        train_result, test_result = pipe.dual_transform(train, test)
        results.append((pipe, train_result, test_result))

    return results


def lrun(clist, train=NoData, test=NoData):
    if not isinstance(clist, ConfigList):
        raise Exception("Exhaustive search is only possible on FiniteCS!")

    for pipe in clist:
        train_result, test_result = pipe.dual_transform(train, test)
        yield pipe, train_result, test_result


def cut(iterable, start=0.75, end=1.0):
    tp = tuple(iterable)
    start, end = int(floor(start * len(tp))), int(ceil(end * len(tp)))

    if start > end:
        raise Exception("The 'start' should be less or equal than 'end'.")
    return ConfigList(components=tp[start:end])


def sort(clist, train=NoData, test=NoData, key=lambda x: (x[1], x[0]), reverse=False):
    """Exhaustive search to maximize value at 'field'.

    Return 'n' best pipelines."""
    return ConfigList(
        components=map(
            _[1], sorted(map(key, lrun(clist, train, test)), reverse=reverse)
        )
    )


def optimize(clist, n=1, train=NoData, test=NoData, better="higher"):
    if not isinstance(clist, ConfigList):
        raise Exception("Exhaustive search is only possible on FiniteCS!")
    higher = "higher"  # TODO: ?? [davi]
    smaller = "smaller"

    select = None
    if better == "higher":
        select = nlargest
    elif better == "smaller":
        select = nsmallest
    else:
        raise ValueError(
            f"Expected '{higher}' or '{smaller}' in 'better', but was given '{better}'"
        )

    def dual(component):
        return component.dual_transform(train, test)[1], component

    return ConfigList(components=map(_[1], select(n, map(dual, clist))))
