from sklearn.cluster import FeatureAgglomeration

# Data reduction by feature agglomeration
from paje.ml.element.preprocessing.unsupervised.feature.transformer.reductor import \
    Reductor

'''
Feature agglomeration

Example:
from paje.preprocessing.data_reduction.DRFtAg import DRFtAg
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

# create a DRFtAg instance
fa = DRFtAg(data)

# apply feature agglomeration to reduce n to 2 collumns
rd = fa.apply(2)
'''


class DRFtAg(Reductor):
    def build_impl(self):
        newconfig = self.config.copy()
        if newconfig['linkage'] == 'ward':
            newconfig['affinity'] = 'euclidean'
        newconfig['n_clusters'] = newconfig.pop('n_components')  # Replace key name.
        self.model = FeatureAgglomeration(**newconfig)

    @classmethod
    def specific_node(cls, data):
        dists = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine', 'precomputed']
        return {'affinity': ['c', dists],
                'linkage': ['c', ['ward', 'complete', 'average', 'single']]}
