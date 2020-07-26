"""
Operating configuration spaces (basic)
======================================

A configuration space can represent the hyperparameters of a single component
(an algorithm) or the hyperparameters of all components contained in the
pipeline.

We represent as a workflow the union of configuration spaces of different
algorithms that together can create a multitude of machine learning pipeline
types.

You can create workflows using the following configuration space operators:

    * Chain -->  It creates a sequential chain of configuration spaces

    * Shuffle --> It shuffles the configuration spaces order

    * Select --> It selects one of the given configuration spaces


"""

################################################################################
# Importing the required packages
#

import numpy as np

from pjautoml.cs.operator.free.chain import Chain
from pjautoml.cs.operator.free.select import Select
from pjautoml.cs.operator.free.shuffle import Shuffle
from pjpy.modeling.supervised.classifier.dt import DT
from pjpy.modeling.supervised.classifier.svmc import SVMC
from pjpy.processing.feature.reductor.pca import PCA
from pjpy.processing.feature.scaler.minmax import MinMax

np.random.seed(0)

################################################################################
# Using `Chain`
# -------------
# The ``Chain`` is a configuration space operator that concatenates other spaces
# in a sequence. Intuitively you can see it as a Cartesian product between two
# or more search spaces.
#

exp = Chain(SVMC, DT)
print(exp.sample())

# You can also use the python operator ``*``

exp = SVMC * DT
print(exp.sample())

################################################################################
# Using `Shuffle`
# ---------------
# The ``Select`` is a configuration space operator that works like a
# bifurcation, where only one of the spaces will be selected. Intuitively you
# can see it as a branch created in your search space in which a random factor
# can enable one or other configuration space.
#

exp = Chain(PCA, MinMax)
print(exp.sample())

################################################################################
# You can also use the python operator ``@``
#

exp = PCA @ MinMax
print(exp.sample())

################################################################################
# Using `Select`
# --------------
#
# The ``Shuffle`` is a configuration space operator that concatenate
# configurations spaces in a sequence, but the order is not maintained.
# Intuitively, you can see it as the union of the Cartesian product of all
# configuration space combinations.
#

exp = Chain(SVMC, DT)
print(exp.sample())

################################################################################
# You can also use the python operator ``+``
#

exp = SVMC + DT
print(exp.sample())

################################################################################
# Using them all:
# ---------------
#
# Using these simple operations, you can create diverse kind of configuration
# spaces to represent an end-to-end AutoML problem.
#

exp = Chain(Shuffle(PCA, MinMax), Select(SVMC + DT))
print(exp.sample())

################################################################################
# You can also use python operators
#

exp = PCA @ MinMax * (SVMC + DT)
print(exp.sample())
