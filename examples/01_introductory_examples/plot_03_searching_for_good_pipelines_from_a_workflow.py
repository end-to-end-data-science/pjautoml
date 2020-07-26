"""
Searching for good pipelines from a workflow
============================================
Let's run a random search to find a good machine learning pipeline to a given
problem.

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
# First, we must create a workflow.
#

workflow = Workflow(
    File("../data/iris.arff"),
    Partition(),
    Map(Shuffle(PCA, MinMax), Select(SVMC + DT), Metric()),
    Summ(function="mean"),
    Reduce(),
    Report("Mean S: $S"),
)

################################################################################
# Now, we run the random search over the workflow created. We will get the best
# pipeline found after 30 samples. The testing performance by pipeline will be
# printed.
#

rs1 = RandomSearch(workflow, sample=30)
print(len(rs1.datas))
print(len(rs1.components))

################################################################################
# The best pipeline found is:
#

res_train, res_test = rs1.datas[0]
print("Train result: ", res_train)
print("test result: ", res_test)
