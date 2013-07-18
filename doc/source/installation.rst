.. _installation:


Introduction
------------

epac depends on scikit-learn and soma-workflow (optionally run on hpc and torque/pbs has been tested).
epac has been tested on python 2.7 so that we recommand that run epac on python 2.7
or its latest version, but not with python 3.0.
In this section, we will present how to install epac on ubuntu and manually on the other platforms.


Ubuntu
------

First of all, you need to install some softwares for epac:


.. code-block:: guess

    sudo apt-get install python-pip
    sudo apt-get install python-setuptools python-dev build-essential libatlas-dev python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose
    pip install --user scikit-learn
    pip install --user soma-workflow


Secondly, epac can be downloaded from github and you can run installation script for your local user.

.. code-block:: guess

    git clone https://github.com/neurospin/pylearn-epac.git
    cd pylearn-epac
    python setup.py install --user


Other platforms
---------------

On the other platforms which support python, you can manually install epac according to your system configuration.

**scikit-learn**: epac depends on scikit-learn which is a manchine learning libary. To use epac, scikit-learn should be installed on your computer. Please goto http://scikit-learn.org/ to install scikit-learn.

**soma-workflow** (optionally): you can install soma-workflow so that epac can run on the hpc (torque/pbs). To install soma-workflow, please goto http://brainvisa.info/soma/soma-workflow for documentation, and https://pypi.python.org/pypi/soma-workflow for installation.

**epac**: download epac from github to ``$EPACDIR`` and set enviroment variable ``$PYTHONPATH`` that contains ``$EPACDIR`` (epac directory), and ``$PATH`` contains $EPACDIR/bin

.. code-block:: guess

    EPACDIR=epac
    git clone https://github.com/neurospin/pylearn-epac.git $EPACDIR
    export PYTHONPATH=$EPACDIR:$PYTHONPATH
    export PATH=$EPACDIR/bin:$PATH


Now, you can start to use epac for machine learning.

