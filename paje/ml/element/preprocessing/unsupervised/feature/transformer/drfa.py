from sklearn.decomposition import FactorAnalysis

# Data reduction by factor analysis
from paje.ml.element.preprocessing.unsupervised.feature.transformer.reductor import Reductor

'''
Factor analysis implementation

Factor analysis is applied as follows:
    - A - A.mean(0) = L @ F + error

    - A is a matrix input with dimension m x n

    - L is an m x k (amount of components) loading matrix

    - F is a k x n common factor matrix

    - error[i] are unobserved stochastic error terms

    - E(F) = 0

    - cov(F) = I

    - cov(A - A.mean(0)) = cov(L @ F + error) = L @ cov(F) @ L.T + cov(error) = L @ L.T + cov(error)

    - A = L @ F + A.mean(0) + error

    - If F[i] is given: p(A[i] | F[i]) = N(L @ F + A.mean(0), cov(error))

    - For a complete probabilistic model we also need a prior distribution for the latent variable F: p(A) = N(A.mean(0), L @ L.T + cov(error))

    - u = A.mean(0)

    - cov(A - u) = L @ L.T + cov(error)

    - C = numpy.cov(A.T) ou C = (1 / (m - 1)) * (A - u).T @ (A - u)

    - p(A) = N(u, C) # multivariate normal distribution

    -

    -

    - F_scores = (A - u).T @ F (será)

    -

    -

    -

    -

    -

    - # numpy.exp(-0.5 * (A - u).T @ numpy.linalg.inv(C) @ (A - u)) / numpy.sqrt(numpy.power(2 * math.pi, len(u)) * numpy.linalg.det(C))

    - u = A.mean(0)

    - Z = (A - A.mean(0)) / A.std(0)

    - C = (1 / (m - 1)) * (Z).T @ (Z)

    - eig_vals, eig_vecs = numpy.linalg.eig(C)

    - L = numpy.diag(eig_vals)

    - loadings = eig_vecs @ numpy.sqrt(L)

    - B = eig_vecs @ numpy.diag(numpy.power(eig_vals, -0.5)) # score matrix

    - F = Z @ B# factor scores

    -

    -

    -


Example:
from paje.preprocessing.data_reduction.DRPCA import DRFA
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

# create a DRFA instance
fa = DRFA(data)

# apply factor analysis to reduce n to 2 collumns
rd = fa.apply(2)
'''


class DRFA(Reductor):
    def build_impl(self):
        self.model = FactorAnalysis(**self.config)

    @classmethod
    def specific_node(cls, data):
        return {}
