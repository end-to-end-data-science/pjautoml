'''
Data reduction methods

Author: Jefferson Tales Oliva
'''

'''
This method is a PCA (principal component analysis) implementation.

- given matrix A with dimension m (instances) x n (features) and d value, which 
is the new amount of features. PCA aims to reduce a matrix Amxn to Amxd

- In this method version, the z-score technique was not implemented


- PCA is applied in following steps:

    1 - feature standardization (PCA is sensible to the measure scale): 
    A = StandardScaler().fit_transform(A)

    2 - Measure the average for each line of the matrix A: u = A.mean(0)

    3 - Measuring covariance: C = (1 / (m - 1)) * (A - u).T @ (A - u)

    4 - Get eigenvalues and eigenvectors: 
    eig_vals, eig_vecs = numpy.linalg.eig(C)

    5 - Get eigenvector subset: W = eig_vecs[:, 0 : d]

    6 - PCA storage: Y = A @ W # Y = A W


- apply_PCA parameters are presented bellow:

    - table (DataFrame): it is a feature dataset, 
    where the lines are instances and columns are attribute-values.

    - features (list): it is a list of feature labels. 
    The i-th feature label must correspond to a the i-th table column

    - class_label (string): it is the label of an attribute used as class 
    (e.g. class, target, etc)

    - n_components (integer): number of features for the redimensioned table

return: redimensioned table of features


Example:
import [main folder name in which data_reduction.py is localized]
.preprocessing.data_reduction as dr 
# Ex.: from AutoGenMR.preprocessing import data_reduction as dr
import pandas as pd

col_names = 
['sepal length', 'sepal width', 'petal length', 'petal width', 'target']
features = col_names[0 : len(col_names) - 1]
cl = str(col_names[len(col_names) - 1])
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# load dataset into Pandas DataFrame
df = pd.read_csv(url, names = col_names)

# redimensioned DataFrame
r_df = dr.apply_PCA(df, features, cl, 2)
'''


def apply_PCA(table, features, label, n_components):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from paje import feature_file_processor

    # get all features and class instances
    x, y = feature_file_processor.split_features_target(table, features, label)

    # standardize features: PCA is sensible to the measure scale
    x = StandardScaler().fit_transform(x)

    # apply PCA
    pca = PCA(n_components=n_components)
    pc = pca.fit_transform(x)

    # generate a feature table and return it
    return feature_file_processor.generate_data_frame(pc, table[[label]])


'''
Factor analysis implementation


Factor analysis is applied as follows:
    
Parameters:

    - table (DataFrame): it is a feature dataset, 
    where the lines are instances and columns are attribute-values.

    - features (list): it is a list of feature labels. 
    The i-th feature label must correspond to a the i-th table column

    - class_label (string): it is the label of an attribute used as class 
    (e.g. class, target, etc)

    - n_components (integer): number of features for the redimensioned table

return: redimensioned table of features


Example:
import [main folder name in which data_reduction.py is localized].preprocessing
.data_reduction as dr # Ex.: from AutoGenMR.preprocessing import data_reduction
 as dr import pandas as pd

col_names = 
['sepal length', 'sepal width', 'petal length', 'petal width', 'target']
features = col_names[0 : len(col_names) - 1]
cl = str(col_names[len(col_names) - 1])
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# load dataset into Pandas DataFrame
df = pd.read_csv(url, names = col_names)

# redimensioned DataFrame
r_df = dr.apply_factor_analysis(df, features, cl, 2)
'''


def apply_factor_analysis(table, features, class_label, n):
    from sklearn.decomposition import FactorAnalysis
    from paje import feature_file_processor

    # get all features and class instances
    x, y = feature_file_processor.split_features_target \
        (table, features, class_label)

    pc = FactorAnalysis(n_components=n, random_state=0).fit_transform(x)

    # generate a feature table and return it
    return feature_file_processor.generate_data_frame(pc, table[[class_label]])


'''
Singular value decomposition


- SVD is applied as follows:

    - Y = USV^T. For data reduction, the trasformation only must be US

    - U = eig(x @ x.T) #U is a m x m orthonormal matrix of 'left-singular' 
    (eigen)vectors of  xx^T

    - lmbV, _ = eig(x.T @ x) #V is a n x n orthonormal matrix of 
    'right-singular' (eigen)vectors of  x^T

    - S = sqrt(diag(abs(lmbV))[:n_components,:]) # S is a m x n diagonal matrix 
    of the square root of nonzero eigenvalues of U or V


- apply_SVD parameters:

    - table (DataFrame): it is a feature dataset, where the lines are instances 
    and columns are attribute-values.

    - features (list): it is a list of feature labels. The i-th feature label
     must correspond to a the i-th table column

    - class_label (string): it is the label of an attribute used as class 
    (e.g. class, target, etc)

    - n_components (integer): number of features for the redimensioned table

return: redimensioned table of features


Example:
import [main folder name in which data_reduction.py is localized].preprocessing
.data_reduction as dr # Ex.: from AutoGenMR.preprocessing import data_reduction as dr
import pandas as pd

col_names = 
['sepal length', 'sepal width', 'petal length', 'petal width', 'target']
features = col_names[0 : len(col_names) - 1]
cl = str(col_names[len(col_names) - 1])
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# load dataset into Pandas DataFrame
df = pd.read_csv(url, names = col_names)

# redimensioned DataFrame
r_df = dr.apply_SVD(df, features, cl, 2)
'''


def apply_SVD(table, features, label, n_components):
    from numpy import diag
    from scipy.sparse.linalg import svds
    from paje import feature_file_processor

    # get all features and class instances
    x, y = feature_file_processor.split_features_target(table, features, label)

    # apply SVD
    u, s, _ = svds(x, n_components)
    # If we use V^T in this operation, the pc will have the original dimension
    pc = u @ diag(s)

    # generate a feature table and return it
    return feature_file_processor.generate_data_frame(pc, table[[label]])


'''
Sparse random projections

- table (DataFrame): it is a feature dataset, where the lines are instances and
 columns are attribute-values.

- features (list): it is a list of feature labels. The i-th feature label must
 correspond to a the i-th table column

- class_label (string): it is the label of an attribute used as class 
(e.g. class, target, etc)

- n_components (integer): number of features for the redimensioned table

return: redimensioned table of features

Example:
import [main folder name in which data_reduction.py is localized].preprocessing
.data_reduction as dr # Ex.: from AutoGenMR.preprocessing import data_reduction
 as dr import pandas as pd

col_names = 
['sepal length', 'sepal width', 'petal length', 'petal width', 'target']
features = col_names[0 : len(col_names) - 1]
cl = str(col_names[len(col_names) - 1])
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# load dataset into Pandas DataFrame
df = pd.read_csv(url, names = col_names)

# redimensioned DataFrame
r_df = dr.apply_SRP(df, features, cl, 2)
'''


def apply_SRP(table, features, label, n_components):
    from sklearn.random_projection import SparseRandomProjection
    from paje import feature_file_processor

    x, y = feature_file_processor.split_features_target(table, features, label)

    rp = SparseRandomProjection \
        (n_components=n_components, dense_output=True, random_state=420)
    pc = rp.fit_transform(x)

    return feature_file_processor.generate_data_frame(pc, table[[label]])


'''
Gaussian random projections

- table (DataFrame): it is a feature dataset, where the lines are instances and 
columns are attribute-values.

- features (list): it is a list of feature labels. The i-th feature label must 
correspond to a the i-th table column

- class_label (string): it is the label of an attribute used as class 
(e.g. class, target, etc)

- n_components (integer): number of features for the redimensioned table

return: redimensioned table of features

Example:
import [main folder name in which data_reduction.py is localized].preprocessing
.data_reduction as dr # Ex.: from AutoGenMR.preprocessing import data_reduction
 as dr import pandas as pd

col_names = 
['sepal length', 'sepal width', 'petal length', 'petal width', 'target']
features = col_names[0 : len(col_names) - 1]
cl = str(col_names[len(col_names) - 1])
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# load dataset into Pandas DataFrame
df = pd.read_csv(url, names = col_names)

# redimensioned DataFrame
r_df = dr.apply_GRP(df, features, cl, 2)
'''


def apply_GRP(table, features, label, n_components):
    from sklearn.random_projection import GaussianRandomProjection
    from AutoGenMR import feature_file_processor
    # from paje import feature_file_processor

    x, y = feature_file_processor.split_features_target(table, features, label)

    rp = GaussianRandomProjection \
        (n_components=n_components, eps=0.1, random_state=420)
    pc = rp.fit_transform(x)

    return feature_file_processor.generate_data_frame(pc, table[[label]])


'''
Feature agglomeration

- table (DataFrame): it is a feature dataset, where the lines are instances and 
columns are attribute-values.

- features (list): it is a list of feature labels. The i-th feature label must 
correspond to a the i-th table column

- class_label (string): it is the label of an attribute used as class 
(e.g. class, target, etc)

- n_components (integer): number of features for the redimensioned table

return: redimensioned table of features

Example:
import [main folder name in which data_reduction.py is localized].preprocessing
.data_reduction as dr # Ex.: from AutoGenMR.preprocessing import data_reduction
 as dr import pandas as pd

col_names = 
['sepal length', 'sepal width', 'petal length', 'petal width', 'target']
features = col_names[0 : len(col_names) - 1]
cl = str(col_names[len(col_names) - 1])
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# load dataset into Pandas DataFrame
df = pd.read_csv(url, names = col_names)

# redimensioned DataFrame
r_df = dr.apply_feature_agglomeration(df, features, cl, 2)
'''


def apply_feature_agglomeration(table, features, label, n_components):
    from sklearn.cluster import FeatureAgglomeration
    from paje import feature_file_processor

    x, y = feature_file_processor.split_features_target(table, features, label)

    fa = FeatureAgglomeration(n_clusters=n_components, linkage='ward')
    pc = fa.fit_transform(x)

    return feature_file_processor.generate_data_frame(pc, table[[label]])


'''
Independent component analysis

- table (DataFrame): it is a feature dataset, where the lines are instances and 
columns are attribute-values.

- features (list): it is a list of feature labels. The i-th feature label must 
correspond to a the i-th table column

- class_label (string): it is the label of an attribute used as class 
(e.g. class, target, etc)

- n_components (integer): number of features for the redimensioned table

return: redimensioned table of features

Example:
import [main folder name in which data_reduction.py is localized].preprocessing
.data_reduction as dr # Ex.: from AutoGenMR.preprocessing import data_reduction
 as dr import pandas as pd

col_names = 
['sepal length', 'sepal width', 'petal length', 'petal width', 'target']
features = col_names[0 : len(col_names) - 1]
cl = str(col_names[len(col_names) - 1])
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# load dataset into Pandas DataFrame
df = pd.read_csv(url, names = col_names)

# redimensioned DataFrame
r_df = dr.apply_ICA(df, features, cl, 2)
'''


def apply_ICA(table, features, label, n_components):
    from sklearn.decomposition import FastICA

    x, y = feature_file_processor.split_features_target(table, features, label)

    ica = FastICA(n_components=n_components, random_state=420)
    pc = ica.fit_transform(x)

    return feature_file_processor.generate_data_frame(pc, table[[label]])
