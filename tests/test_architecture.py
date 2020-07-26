"""Module dedicated to framework testing."""
import numpy as np

from pjautoml.cs.operator.datadriven.optimization.modelfree.best import Best
from pjautoml.cs.operator.datadriven.optimization.modelfree.random import RandomSearch
from pjautoml.cs.operator.free.chain import Chain
from pjautoml.cs.operator.free.map import Map
from pjautoml.cs.operator.free.sample import Sample
from pjautoml.cs.operator.free.select import Select
from pjautoml.cs.operator.free.shuffle import Shuffle
from pjautoml.cs.workflow import Workflow
from pjml.data.evaluation.metric import Metric
from pjml.data.flow.file import File
from pjml.operator.pipeline import Pipeline
from pjml.stream.expand.partition import Partition
from pjml.stream.reduce.reduce import Reduce
from pjml.stream.reduce.summ import Summ
from pjpy.modeling.supervised.classifier.dt import DT
from pjpy.modeling.supervised.classifier.svmc import SVMC
from pjpy.processing.feature.reductor.pca import PCA
from pjpy.processing.feature.scaler.minmax import MinMax


class TestArchitecture:
    """Tests for the framework architecture."""

    @staticmethod
    def workflow_with_python_operators():
        return (
                File("tests/test_datasets/num_Iris.arff")
                * Partition()
                * Map((PCA @ MinMax) * (SVMC + DT) * Metric())
                * Summ()
                * Reduce()
        )

    @staticmethod
    def workflow():
        return Workflow(
            File("tests/test_datasets/num_Iris.arff"),
            Partition(),
            Map(Chain(Shuffle(PCA, MinMax), Select(SVMC + DT), Metric())),
            Summ(),
            Reduce(),
        )

    @staticmethod
    def workflow_ml_automl():
        return (
                File("tests/test_datasets/num_Iris.arff")
                * Partition()
                * Map(Pipeline(PCA(), SVMC()) + ((PCA @ MinMax) * (SVMC + DT + DT(criterion="gini")) * Metric()))
                * Summ()
                * Reduce()
        )

    @staticmethod
    def random_search(workflow):
        np.random.seed(0)

        n = 5
        rs = RandomSearch(workflow, sample=n)
        size_datas = len(rs.datas)
        size_components = len(rs.components)

        assert size_datas == 1
        assert size_components == 1

    def test_random_search(self):
        np.random.seed(0)
        self.random_search(self.workflow_with_python_operators())

    def test_random_search_with_python_operators_(self):
        np.random.seed(0)
        self.random_search(self.workflow())

    def test_random_search_operators(self):
        np.random.seed(0)

        workflow = self.workflow()

        n = 5
        spl = Sample(workflow, n=n)
        size = len(spl)
        assert size == n

        n = 1
        best_2 = Best(spl, n=n)
        size_datas = len(best_2.datas)
        size_components = len(best_2.components)

        assert size_datas == n
        assert size_components == n
