from paje.automl.composer.composer import Composer
from paje.base.chain import Chain
from paje.searchspace.hp import CatHP
from paje.searchspace.configspace import ConfigSpace


class Iterator(Composer):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.iterable = self.materialize(self.config['iterable'])
        self.field = self.config['field']

    def apply_impl(self, data):
        component = self.components[0]
        self.model = []

        splits = self.iterable.iterations(data)
        chain = data.C
        idx = 0
        for split in splits:
            train_data = split.apply(data)
            component.apply(train_data)
            aux = component.use(train_data)
            if aux is None:
                break
            chain = Chain(aux.get(self.field), chain, idx=idx)

            self.model.append((split, component))
            component = self.materialize(component.config)
            idx += 1

        return data.updated(self, C=chain)

    def use_impl(self, data):
        """ This function will be called by Component in the the 'use()' step.

        Attributes
        ----------
        data: :obj:`Data`
            The `Data` object that represent a dataset used for testing phase.
        """

        chain = data.C
        idx = 0
        for split, component in self.model:
            test_data = split.use(data)
            aux = component.use(test_data)
            if aux is None:
                break

            chain = Chain(aux.get(self.field), chain, idx=idx)

            if component.failed:
                raise Exception('Using subcomponent failed! ', component)

            idx += 1
        return data.updated(self, C=chain)

    @classmethod
    def cs_impl(cls, config_spaces):
        hps = [
            CatHP('configs', cls.sampling_function,
                  config_spaces=config_spaces[0]),
            CatHP('reduce', cls.sampling_function,
                  config_spaces=config_spaces[1])
        ]
        return ConfigSpace(name=cls.__name__, hps=hps)

    @staticmethod
    def sampling_function(config_spaces):
        raise Exception('useless call!!!!!!!!')

    def modifies(self, op):
        return ['C']
