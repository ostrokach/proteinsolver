#!/bin/bash

set -ev

rsync -av \
    --include="*/" --include="*.svg" --include="*.png" --include="*.pdf" --exclude="*" \
    --prune-empty-dirs \
    ./notebooks/ ./public/figures/
