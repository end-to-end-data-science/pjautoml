import hashlib
import json

import numpy
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, \
    StratifiedShuffleSplit

from paje.base.iterable import Iterable
from paje.ml.element.preprocessing.split import Split


class CV(Iterable):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self._memoized = {}
        self._max = 0
        # split, steps, test_size, random_state
        if self.config['split'] == "cv":
            self.model = StratifiedKFold(
                shuffle=True,
                n_splits=self.config['steps'],
                random_state=self.config['random_state'])
        elif self.config['split'] == "loo":
            self.model = LeaveOneOut()
        elif self.config['split'] == 'holdout':
            self.model = StratifiedShuffleSplit(
                n_splits=self.config['steps'],
                test_size=self.config['test_size'],
                random_state=self.config['random_state'])
        self.fields = self.config['fields']

    def iterations(self, data):
        lst = []
        zeros = numpy.zeros(data.get_matrix(self.fields[0]).shape[0])
        partitions = list(self.model.split(X=zeros, y=zeros))
        for it, (tr_idxs, ts_idxs) in enumerate(partitions):
            txt = json.dumps([tr_idxs.tolist(), ts_idxs.tolist()]).encode()
            split = Split(
                config={
                    'fields': self.fields,
                    'iteration': it,
                    'hash': hashlib.md5(txt).hexdigest()
                },
                train=tr_idxs,
                test=ts_idxs
            )
            lst.append(split)
        return lst

    @classmethod
    def cs_impl(cls):
        raise NotImplementedError('Please implement me! Dont forget "fields" '
                                  'SubHP for subsets of a set of categoricals')
        # config_space = ConfigSpace('CV')
        # start = config_space.start()
        #
        # node = config_space.node()
        # start.add_child(node)
        # node.add_hp(CatHP('iteration', choice, a=[0]))
        #
        # holdout = config_space.node()
        # node.add_child(holdout)
        # holdout.add_hp(CatHP('split', choice, a=['holdout']))
        # holdout.add_hp(IntHP('steps', choice, low=1, high=100000))
        # holdout.add_hp(RealHP('test_size', choice, low=1e-06, high=1 - 1e-06))
        #
        # cv = config_space.node()
        # node.add_child(cv)
        # cv.add_hp(CatHP('split', choice, a=['cv']))
        # cv.add_hp(IntHP('steps', choice, low=1, high=100000))
        #
        # loo = config_space.node()
        # node.add_child(loo)
        # loo.add_hp(CatHP('split', choice, a=['loo']))
        #
        # return config_space

    def modifies(self, op):
        from paje.base.data import Data
        return Data.all_mats
