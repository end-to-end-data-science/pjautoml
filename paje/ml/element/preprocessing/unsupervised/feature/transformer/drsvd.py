from numpy import diag
from scipy.sparse.linalg import svds

# Data reduction by SVD
from paje.ml.element.preprocessing.unsupervised.feature.transformer.reductor \
    import Reductor

'''
Singular value decomposition

- SVD is applied as follows:

    - Y = USV^T. For data reduction, the trasformation only must be US

    - U = eig(x @ x.T) #U is a m x m orthonormal matrix of 'left-singular' (eigen)vectors of  xx^T

    - lmbV, _ = eig(x.T @ x) #V is a n x n orthonormal matrix of 'right-singular' (eigen)vectors of  x^T

    - S = sqrt(diag(abs(lmbV))[:n_components,:]) # S is a m x n diagonal matrix of the square root of nonzero eigenvalues of U or V


Example:
from paje.preprocessing.data_reduction.DRSVD import DRSVD
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

# create a DRSVD instance
svd = DRSVD(data)

# apply svd to reduce n to 2 collumns
rd = svd.apply(2)
'''


class DRSVD(Reductor):
    def build_impl(self):
        raise Exception("this module is broken")
        # TODO: this module is broken

    def apply_impl(self, data):
        u, s, _ = svds(data.X, **self.config)
        # If we use V^T in this operation, pc will have the original dimension
        self.model = u @ diag(s)
        return data.updated(self, X=self.model)  # TODO: this module is broken

    @classmethod
    def specific_node(cls, data):
        return {}
