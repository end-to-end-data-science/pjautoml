from heapq import nlargest, nsmallest

from pjautoml.cs.operand.list.flist import CList, ListCS
from pjdata.aux.util import _
from pjdata.content.specialdata import NoData


# def best(clist):
#     return Best(clist)


class Best(CList):
    def __init__(self, listcs, n=1, train=NoData, test=NoData, better="higher"):
        if not isinstance(listcs, ListCS):
            raise Exception("Exhaustive search is only possible on finite"
                            "configuration space (FCS)!")
        select = None
        if better == "higher":
            select = nlargest
        elif better == "smaller":
            select = nsmallest

        self.n = n
        self.train = train
        self.test = test
        self.better = better

        def dual(singleton):
            return singleton.component.dual_transform(self.train, self.test), singleton.component

        select = select(n, map(dual, listcs), key=lambda x: x[0][1])
        self.datas = tuple(map(_[0], select))
        self.components = tuple(map(_[1], select))

        super().__init__(*self.components)
