#!/bin/bash

set -ev

python -m pytest \
    -c setup.cfg \
    --color=yes \
    --cov="${SP_DIR}/proteinsolver" \
    --cov-report html \
    --junitxml=pytest.xml \
    tests/

ls -al

sed -i "s|${SP_DIR}/||g" .coverage
mv .coverage "${RECIPE_DIR}/.coverage"
mv pytest.xml "${RECIPE_DIR}/pytest.xml"
mv htmlcov "${RECIPE_DIR}/htmlcov"
