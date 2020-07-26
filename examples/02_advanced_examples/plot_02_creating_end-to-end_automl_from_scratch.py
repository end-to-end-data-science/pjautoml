"""
Creating an end-to-end AutoML from scratch
==========================================
Let's create an AutoML from scratch.

"""

################################################################################
# Importing the required packages
#

import numpy as np

from pjautoml.cs.operator.datadriven.optimization.modelfree.random import RandomSearch
from pjautoml.cs.operator.free.map import Map
from pjautoml.cs.operator.free.select import Select
from pjautoml.cs.operator.free.shuffle import Shuffle
from pjautoml.cs.workflow import Workflow
from pjml.data.communication.report import Report
from pjml.data.evaluation.metric import Metric
from pjml.data.flow.file import File
from pjml.stream.expand.partition import Partition
from pjml.stream.reduce.reduce import Reduce
from pjml.stream.reduce.summ import Summ
from pjpy.modeling.supervised.classifier.dt import DT
from pjpy.modeling.supervised.classifier.svmc import SVMC
from pjpy.processing.feature.reductor.pca import PCA
from pjpy.processing.feature.scaler.minmax import MinMax

np.random.seed(0)


################################################################################
# First, we can define a workflow. Notice we not add a ``File``. Then, we use
# random search as the optimization process to select the best pipeline.
# Finally, we should also give it a name. The name of my AutoML, of course, is
# ``my_automl`` :)
#


def my_automl(data):
    workflow = Workflow(
        Partition(),
        Map(Shuffle(PCA, MinMax), Select(SVMC + DT), Metric()),
        Summ(function="mean"),
        Reduce(),
        Report("Mean S: $S"),
    )

    rs = RandomSearch(workflow, sample=30, train=data, test=data)
    return rs.components[0]


################################################################################
# Now, let's find a good pipeline for the iris dataset:
#

data = File("../data/iris.arff").data
best_pipeline = my_automl(data)
print(best_pipeline)
