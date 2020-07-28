.. _sphx_api:

#################
API Documentation
#################
This is the full API documentation of the `pjautoml` package.

.. _cs_ref:

:mod:`pjautoml.cs`: Configuration Space
=======================================

.. automodule:: pjautoml.cs
    :no-members:
    :no-inherited-members:

.. currentmodule:: pjautoml


.. _cs_operand_ref:

Operand
-------

.. automodule:: pjautoml.cs.operand
   :no-members:
   :no-inherited-members:

.. currentmodule:: pjautoml.cs.operand

.. autosummary::
   :toctree: generated/

   graph.graph.Graph
   graph.node.Node
   list.flist.ListCS
   list.flist.CList
   list.flist.FList


.. _cs_operator_ref:

Operator
--------

.. automodule:: pjautoml.cs.operator
   :no-members:
   :no-inherited-members:

.. currentmodule:: pjautoml.cs.operator

.. autosummary::
   :toctree: generated/


.. _cs_operator_data-driven_ref:

Data-driven configuration space operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pjautoml.cs.operator.datadriven
   :no-members:
   :no-inherited-members:

.. currentmodule:: pjautoml.cs.operator.datadriven

.. autosummary::
   :toctree: generated/

    optimization.modelfree.best.Best
    optimization.modelfree.random.RandomSearch

Configuration space operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pjautoml.cs.operator.free
   :no-members:
   :no-inherited-members:

.. currentmodule:: pjautoml.cs.operator.free

.. autosummary::
   :toctree: generated/

    container.Container
    map.Map
    multi.Multi
    sample.Sample
    chain.Chain
    select.Select
    shuffle.Shuffle


.. _util_ref:

:mod:`pjautoml.util`: Util Classes and Functions
================================================
.. automodule:: pjautoml.util
    :no-members:
    :no-inherited-members:

.. currentmodule:: pjautoml.util

.. autosummary::
   :toctree: generated/

    parameter.Param
    parameter.CatP
    parameter.IntP
    parameter.FixedP
    parameter.OrdP
    parameter.RealP
..    parameter.SubP
..    parameter.PermP


.. _abs_ref:

:mod:`pjautoml.abs`: Abstract Classes and Mixin
===============================================

.. automodule:: pjautoml.abs
    :no-members:
    :no-inherited-members:

.. currentmodule:: pjautoml.abs

.. autosummary::
   :toctree: generated/

   mixin.asoperand.AsOperandCS
..   component.Component

