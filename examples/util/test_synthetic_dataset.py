import numpy as np
from paje.util.synthetic.dataset import ClassificationDataset

def test_classification_dataset():
    X_ = np.array([[0.5488135 , 0.71518937, 0.60276338, 0.54488318],
                   [0.4236548 , 0.64589411, 0.43758721, 0.891773  ],
                   [0.96366276, 0.38344152, 0.79172504, 0.52889492],
                   [0.56804456, 0.92559664, 0.07103606, 0.0871293 ],
                   [0.0202184 , 0.83261985, 0.77815675, 0.87001215],
                   [0.97861834, 0.79915856, 0.46147936, 0.78052918],
                   [0.11827443, 0.63992102, 0.14335329, 0.94466892],
                   [0.52184832, 0.41466194, 0.26455561, 0.77423369],
                   [0.45615033, 0.56843395, 0.0187898 , 0.6176355 ]])
    y_ = np.array([1, 1, 2, 0, 2, 2, 0, 1, 0])

    cd = ClassificationDataset(n_class=3, n_attr=4, n_sample=9, random_state=0)
    X, y = cd.new_dataset()

    assert X.shape == X_.shape
    assert y.shape == y_.shape
    assert np.allclose(X, X_)
    assert np.allclose(y, y_)

