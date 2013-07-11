#! bin/sh

FILENAME=epac
BRANCHNAME=documents

git clone git@github.com:neurospin/pylearn-epac.git $FILENAME
cd $FILENAME
git fetch origin
git checkout -b $BRANCHNAME origin/$BRANCHNAME

# Installation script:
sudo apt-get install python-pip

sudo apt-get install python-setuptools python-dev build-essential libatlas-dev python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose

pip install --user scikit-learn
pip install --user soma-workflow

python setup.py install --user

