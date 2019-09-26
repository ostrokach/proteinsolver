#!/bin/bash

set -ev

python -m pytest \
    -c setup.cfg \
    --cov="${SP_DIR}/proteinsolver" \
    --color=yes \
    --junitxml=pytest.xml \
    tests/

sed -i "s|${SP_DIR}/||g" .coverage
mv .coverage "${RECIPE_DIR}/.coverage"
mv pytest.xml "${RECIPE_DIR}/pytest.xml"
