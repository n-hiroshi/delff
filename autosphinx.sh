#!/bin/bash
# Rename package directory from phct ro sphg
# initial build docs
sphinx-apidoc -F -o docs/ delff
cp conf.py docs/
sphinx-build -b html docs/ docs/_build/
# Rebuild docs
# sphinx-build -b html docs/ docs/_build/
