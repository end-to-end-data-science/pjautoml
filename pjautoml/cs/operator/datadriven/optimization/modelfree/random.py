from pjautoml.cs.operand.list.flist import FList
from pjautoml.cs.operator.datadriven.optimization.modelfree.best import Best
from pjautoml.cs.operator.free.sample import Sample
from pjdata.content.specialdata import NoData


# def rs(cs, sample=100, best=1, train=NoData, test=NoData):
#     return RandomSearch(cs, sample=sample, best=best, train=train, test=test)


class RandomSearch(FList):
    def __init__(self, cs, sample=100, best=1, train=NoData, test=NoData):
        self.sample = sample
        self.best = best
        self.train = train
        self.test = test
        self.cs = cs
        self.sample = Sample(self.cs, n=self.sample)

        best = Best(self.sample, n=self.best, train=self.train, test=self.test)
        self.datas = best.datas
        self.components = best.components

        super().__init__(*best)


# # TODO: Se tranfigura no melhor pipeline ...
# # A ideia de transfigurar sera interessante para
# class CRandomSearch(Component):
#     def __init__(self,
#                  cs,
#                  sample_size=100,
#                  best=1,
#                  train=None,
#                  test=None,
#                  enhance: bool = True,
#                  model: bool = True,
#                  deterministic: bool = False,
#                  nodata_handler: bool = False):
#         config = {'sample': sample_size, 'best': best}
#         super().__init__(config=config, enhance=enhance, model=model, deterministic=deterministic,
#                          nodata_handler=nodata_handler)
#
#         self.cs = cs
#         self.sample_size = sample_size
#         self.best = best
#         self.train = train
#         self.test = test
#
#     def sample(self):
#         return Best(Sample(self.cs, n=self.sample), n=self.best, train=self.train, test=self.test)
#
#     @classmethod
#     def _cs_impl(cls) -> CS:
#         pass
#
#     def _info(self, data):
#         pass
#
#     def _enhancer_info(self, data: t.Data) -> Dict[str, Any]:
#         pass
#
#     def _model_info(self, data: t.Data) -> Dict[str, Any]:
#         pass
#
#     def _enhancer_func(self) -> t.Transformation:
#         pass
#
#     def _model_func(self, data: t.Data) -> t.Transformation:
#         pass
