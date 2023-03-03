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
    echo ""
    echo "Dataset dir 'asset' already exists. Skipping..."
    echo ""
fi

turkcorpus_dir="$BASE/data/turkcorpus"
if [[  ! -d "$turkcorpus_dir"  ]]; then
    git clone https://github.com/cocoxu/simplification.git $turkcorpus_dir
else
    echo ""
    echo "Dataset dir 'turkcorpus' already exists. Skipping..."
    echo ""
fi

hsplit_dir="$BASE/data/hsplit"
if [[  ! -d "$hsplit_dir"  ]]; then
    git clone https://github.com/eliorsulem/HSplit-corpus.git $hsplit_dir
else
    echo ""
    echo "Dataset dir 'hsplit' already exists. Skipping..."
    echo ""
fi

onestopenglish_dir="$BASE/data/onestopenglish"
if [[  ! -d "$onestopenglish_dir"  ]]; then
    git clone https://github.com/nishkalavallabhi/OneStopEnglishCorpus.git $onestopenglish_dir
else
    echo ""
    echo "Dataset dir 'onestopenglish' already exists. Skipping..."
    echo ""
fi