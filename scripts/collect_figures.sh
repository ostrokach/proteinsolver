#!/bin/bash

set -ev

rsync -av \
    --exclude=".ipynb_checkpoints" --exclude="static/" --include="*/" --include="*.svg" --include="*.png" --include="*.pdf" --exclude="*" \
    --prune-empty-dirs --delete \
    ./notebooks/ ./docs/images/

