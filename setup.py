import os
from setuptools import setup
import os.path as op

commands = [op.join('bin', 'epac_mapper')]

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="epac",
    version="0.0.4",
    author="Check contributors on https://github.com/neurospin/pylearn-epac",
    author_emai="edouard.duchesnay@cea.fr",
    description=("Embarrassingly Parallel Array Computing: EPAC is a machine learning workflow builder."),
    license="To define",
    keywords="machine learning, cross validation, permutation, parallel computing",
    url="https://github.com/neurospin/pylearn-epac",
    package_dir={'': './'},
    packages=['epac',
              'epac.map_reduce',
              'epac.sklearn_plugins',
              'epac.workflow'],
    scripts=commands,
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Machine learning"
    ],
    extras_require={
        'machine_learning': ['sklearn']
    },
)