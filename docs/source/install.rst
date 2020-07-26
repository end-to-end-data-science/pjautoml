Install
#######

Requirements
=============

The `pjautoml` package requires the following dependencies:

* numpy
* scipy
* pjml


Install
=======

The `pjautoml` is available on the `PyPi <https://pypi.org/project/pjautoml/>`_
. You can install it via `pip` as follow::

  pip install -U pjautoml


It is possible to use the development version installing from GitHub::
  
  pip install -U git@github.com:end-to-end-data-science/pjautoml.git

  
If you prefer, you can clone it and run the `setup.py` file. Use the following
commands to get a copy from Github and install all dependencies::

  git clone git@github.com:end-to-end-data-science/pjautoml.git
  cd pjautoml
  pip install .


Test and coverage
=================

If you want to test/test-coverage the code before to install::

  $ make install-dev
  $ make test-cov

Or::

  $ make install-dev
  $ pytest --cov=pjautoml/ tests/

