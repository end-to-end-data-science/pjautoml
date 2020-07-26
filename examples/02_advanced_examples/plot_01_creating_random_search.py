"""
Creating random search with configuration spaces' operations
============================================================
Let's create a random search from scratch using the other two essential
configuration space operation: ``Best`` and ``Sample``.

"""

################################################################################
# Importing the required packages
#

import numpy as np

from pjautoml.cs.operator.datadriven.optimization.modelfree.best import Best
from pjautoml.cs.operator.free.map import Map
from pjautoml.cs.operator.free.sample import Sample
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
# This is the workflow we will work on.
# The workflow is also a representation of our configuration space, i.e., it
# represents all machine learning pipelines that can be achieved.
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
# Using `Sample`:
# The operation ``Sample`` will sample ``n`` different pipelines.
# It will transform infinity configuration space in finite.
#

spl = Sample(workflow, n=10)
print(len(spl))

################################################################################
# Using `Best`:
# The operation ``Best`` will return the pipeline with the best performance
# (highest value in ``data.S``). By default, we define the best as the highest
# value, but you can also set the best as the lowest value.
#

best_2 = Best(spl, n=2)
print(len(best_2.datas))
print(len(best_2.components))
