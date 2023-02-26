#!/usr/bin/env bash
# -*- coding: utf-8 -*-

set -e

SCRIPTS_DIR=$(cd $(dirname -- $0); pwd)
BASE="$SCRIPTS_DIR/.."

cd $BASE

asset_dir="$BASE/data/asset"
if [[  ! -d "$asset_dir"  ]]; then
    git clone https://github.com/facebookresearch/asset.git $asset_dir
else
    echo "Dataset dir 'asset' already exists. Skipping..."
fi