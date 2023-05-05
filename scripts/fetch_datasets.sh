#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# __Author__ = 'Tannon Kew'
# __Email__ = 'kew@cl.uzh.ch
# __Date__ = '2023-03-03'

set -e

SCRIPTS_DIR=$(cd $(dirname -- $0); pwd)
BASE="$SCRIPTS_DIR/.."
DATA_DIR="$BASE/resources/data"

cd $BASE

asset_dir="$DATA_DIR/asset"
if [[  ! -d "$asset_dir"  ]]; then
    git clone https://github.com/facebookresearch/asset.git $asset_dir
else
    echo ""
    echo "Dataset dir 'asset' already exists. Skipping..."
    echo ""
fi

turkcorpus_dir="$DATA_DIR/turkcorpus"
if [[  ! -d "$turkcorpus_dir"  ]]; then
    git clone https://github.com/cocoxu/simplification.git $turkcorpus_dir
else
    echo ""
    echo "Dataset dir 'turkcorpus' already exists. Skipping..."
    echo ""
fi

hsplit_dir="$DATA_DIR/hsplit"
if [[  ! -d "$hsplit_dir"  ]]; then
    git clone https://github.com/eliorsulem/HSplit-corpus.git $hsplit_dir
else
    echo ""
    echo "Dataset dir 'hsplit' already exists. Skipping..."
    echo ""
fi

onestopenglish_dir="$DATA_DIR/onestopenglish"
if [[  ! -d "$onestopenglish_dir"  ]]; then
    git clone https://github.com/nishkalavallabhi/OneStopEnglishCorpus.git $onestopenglish_dir
else
    echo ""
    echo "Dataset dir 'onestopenglish' already exists. Skipping..."
    echo ""
fi


plainenglishlegal_dir="$DATA_DIR/plainenglishlegal"
if [[  ! -d "$plainenglishlegal_dir"  ]]; then
    git clone https://github.com/lauramanor/legal_summarization $plainenglishlegal_dir
else
    echo ""
    echo "Dataset dir 'plainenglishlegal' already exists. Skipping..."
    echo ""
fi


contractdata_dir="$DATA_DIR/contractbm"
if [[  ! -d "$contractdata_dir"  ]]; then
    wget https://dax-cdn.cdn.appdomain.cloud/dax-split-and-rephrase/1.0.0/split-and-rephrase-data.tar.gz 
    tar -xvf split-and-rephrase-data.tar.gz
else
    echo ""
    echo "Dataset dir 'contractbm' already exists. Skipping..."
    echo ""
fi