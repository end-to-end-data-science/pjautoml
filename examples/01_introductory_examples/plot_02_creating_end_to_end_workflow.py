"""
Creating an end-to-end workflow
===============================
Let create an end-to-end machine learning workflow.

"""

################################################################################
# Importing the required packages
#

import numpy as np

from pjautoml.cs.operator.free.chain import Chain
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
# First, we create a machine learning expression.
#

exp = Chain(Shuffle(PCA, MinMax), Select(SVMC + DT))

################################################################################
# It represents a configuration space. Let's get a sample from it.
#

print(exp.sample())

################################################################################
# Defined our machine learning expression, we will create an end-to-end
# workflow.
#

workflow = Workflow(
    File("../data/iris.arff"),
    Partition(),
    Map(exp, Metric()),
    Summ(function="mean"),
    Reduce(),
    Report("Mean S: $S"),
)

################################################################################
# or using only python operators
#

workflow = (
    File("../data/iris.arff")
    * Partition()
    * Map(exp * Metric())
    * Summ(function="mean")
    * Reduce()
    * Report("Mean S: $S")
)

################################################################################
# This workflow represents the union of all configuration spaces.
# Let get a sample of it:
#

spl = workflow.sample()
print(spl)
