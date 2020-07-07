import numpy as np


class ClassificationDataset(object):
    """  """

    def __init__(self, n_class=3, n_attr=4, n_sample=500, random_state=None):
        self.n_class = n_class
        self.n_attr = n_attr
        self.n_sample = n_sample

        if random_state != None:
            np.random.seed(random_state)

    # TODO: there is a synthetic dataset generator inside Data()
    def new_dataset(self): 
        X = np.random.rand(self.n_sample, self.n_attr)
        aux = np.argsort(np.sum(X, axis=1))
        idx = np.array_split(aux, self.n_class)

        count = 0
        y = np.zeros_like(aux)
        cat = np.arange(0, self.n_class)

        # TODO: improve this
        for i in range(0, self.n_class):
            y[idx[i]] = i

        return X, y



