#!/bin/bash

# This script has only been test on ubuntu

make html

outdir="$(mktemp -d)"
curdir="$(pwd)"

git clone git@github.com:neurospin/pylearn-epac.git $outdir

cd $outdir
git fetch origin
git checkout -b gh-pages origin/gh-pages
cp -r $curdir/_build/html/* $outdir
git add .
git commit -a -m "DOC: update pages"
git push origin gh-pages
cd $curdir
