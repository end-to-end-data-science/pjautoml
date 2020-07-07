from sklearn.random_projection import GaussianRandomProjection

# Data reduction by GRP
from paje.ml.element.preprocessing.unsupervised.feature.transformer.reductor import Reductor

'''
This class is a Gaussian random projections implementation for data reduction.


Example:
from paje.preprocessing.data_reduction.DRGRP import DRSRP
from paje.data.data import Data
import pandas as pd

col_names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'target']
features = col_names[0 : len(col_names) - 1]
cl = str(col_names[len(col_names) - 1])
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# load dataset into Pandas DataFrame
df = pd.read_csv(url, names = col_names)

# create a Data type instance
x = df.loc[:, features].values
y = df.loc[:,['target']].values
data = Data(x, y)

# create a DRGRP instance
grp = DRGRP(data)

# apply GRP to reduce n to 2 collumns
rd = grp.apply(2)
'''


class DRGRP(Reductor):
    def build_impl(self):
        self.model = GaussianRandomProjection(**self.config)

    @classmethod
    def specific_node(cls, data):
        return {'eps': ['o', [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]}
