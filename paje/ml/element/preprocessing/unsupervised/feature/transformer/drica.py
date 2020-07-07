from sklearn.decomposition import FastICA

# Data reduction by ICA
from paje.ml.element.preprocessing.unsupervised.feature.transformer.reductor import Reductor

'''
This class is a ICA (independent component analysis) implementation for data reduction.

- given matrix A with dimension m (instances) x n (features) and d value, which is the new amount of features. PCA aims to reduce a matrix Amxn to Amxd

- In this method version, the z-score technique was not implemented



Example:
from paje.preprocessing.data_reduction.DRPCA import DRPCA
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

# create a DRICA instance
ica = DRICA(data)

# apply pca to reduce n to 2 collumns
rd = ica.apply(2)
'''


class DRICA(Reductor):
    def build_impl(self):
        self.model = FastICA(**self.config)

    @classmethod
    def specific_node(cls, data):
        return {}
