from paje.base.noniterable import NonIterable


class Split(NonIterable):
    def __init__(self, config, train, test, **kwargs):
        super().__init__(config, **kwargs)
        self.fields = config['fields']
        # from config --> aa'iteration'
        # from config --> 'uuid'

        self.train = train
        self.test = test

    def _core(self, data, idxs):
        new_dic = {f: data.get_matrix(f)[idxs] for f in self.fields}
        return data.updated(self, **new_dic)

    def apply_impl(self, data):
        return self._core(data, self.train)

    def use_impl(self, data):
        return self._core(data, self.test)

    @classmethod
    def cs_impl(cls, **kwargs):
        raise Exception('Split is not for external use!')

    def modifies(self, op):
        return self.fields
