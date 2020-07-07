from paje.element.preprocessing.supervised.feature.selector.statistical.cfs import FilterCFS
from paje.element.preprocessing.supervised.feature.selector.statistical.chi_square import FilterChiSquare
from paje.element.preprocessing.supervised.feature.selector.statistical import FilterFScore
from paje.element.preprocessing.supervised.feature.selector.statistical.gini_index import FilterGiniIndex
from paje.element.preprocessing.supervised.feature.selector.statistical.t_score import FilterTScore
from skfeature.function.statistical_based import \
    chi_square, CFS, f_score, gini_index, t_score
from paje.util.synthetic.dataset import ClassificationDataset
import numpy as np

class TestStatisticalBasedFilters():

    DATA = ClassificationDataset(n_class=3, n_attr=10, n_sample=200,
                                           random_state=0).new_dataset()

    def test_chi_squared(self):
        X, y = self.DATA

        f = FilterChiSquare(ratio=0.5)
        f.fit(X, y)
        X_, y_ = f.transform(X, y)

        score = chi_square.chi_square(X, y)
        rank = chi_square.feature_ranking(score)
        selected = rank[0:5]

        assert f.fit(X, y) is f
        assert np.array_equal(f.rank(), rank)
        assert np.allclose(f.score(), score)
        assert np.array_equal(f.selected(), selected)
        assert np.allclose(X_, X[:,selected])
        assert np.array_equal(y_, y)


    def test_cfs(self):
        X, y = self.DATA

        f = FilterCFS()
        f.fit(X, y)
        X_, y_ = f.transform(X, y)

        score = None
        rank = None
        selected = CFS.cfs(X,y)

        assert f.fit(X, y) is f
        assert f.score() == score
        assert f.rank() == rank
        assert np.array_equal(f.selected(), selected)
        assert np.allclose(X_, X[:,selected])
        assert np.array_equal(y_, y)


    def test_f_score(self):
        X, y = self.DATA

        f = FilterFScore(ratio=0.5)
        f.fit(X, y)
        X_, y_ = f.transform(X, y)

        score = f_score.f_score(X, y)
        rank = f_score.feature_ranking(score)
        selected = rank[0:5]

        assert f.fit(X, y) is f
        assert np.array_equal(f.rank(), rank)
        assert np.allclose(f.score(), score)
        assert np.allclose(X_, X[:,selected])
        assert np.array_equal(y_, y)


    def test_gine_index(self):
        X, y = self.DATA

        f = FilterGiniIndex(ratio=0.5)
        f.fit(X, y)
        X_, y_ = f.transform(X, y)

        score = gini_index.gini_index(X, y)
        rank = gini_index.feature_ranking(score)
        selected = rank[0:5]

        assert f.fit(X, y) is f
        assert np.array_equal(f.rank(), rank)
        assert np.allclose(f.score(), score)
        assert np.allclose(X_, X[:,selected])
        assert np.array_equal(y_, y)


    def test_t_score(self):
        # Binary test
        X, y = ClassificationDataset(n_class=2, n_attr=10, n_sample=200,
                                           random_state=0).new_dataset()

        f = FilterTScore(ratio=0.5)
        f.fit(X, y)
        X_, y_ = f.transform(X, y)

        score = t_score.t_score(X, y)
        rank = t_score.feature_ranking(score)
        selected = rank[0:5]

        assert f.fit(X, y) is f
        assert np.array_equal(f.rank(), rank)
        assert np.allclose(f.score(), score)
        assert np.allclose(X_, X[:,selected])
        assert np.array_equal(y_, y)

        # Multclass test
        X, y = self.DATA
        f = FilterTScore(ratio=0.5)
        f.fit(X, y)
        X_, y_ = f.transform(X, y)

        score1 = t_score.t_score(X[np.any([y==0, y==1], axis=0)], y[np.any([y==0, y==1], axis=0)])
        score2 = t_score.t_score(X[np.any([y==0, y==2], axis=0)], y[np.any([y==0, y==2], axis=0)])
        score3 = t_score.t_score(X[np.any([y==2, y==1], axis=0)], y[np.any([y==2, y==1], axis=0)])
        score = np.sum([score1, score2, score3], axis=0)
        rank = t_score.feature_ranking(score)
        selected = rank[0:5]

        assert f.fit(X, y) is f
        assert np.array_equal(f.rank(), rank)
        assert np.allclose(f.score(), score)
        assert np.allclose(X_, X[:,selected])
        assert np.array_equal(y_, y)


