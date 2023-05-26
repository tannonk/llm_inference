#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# __Author__ = 'Sweta Agrawal'
# __Email__ = 'sweagraw@umd.edu'
# __Date__ = '2023-05-26'

git clone git@github.com:facebookresearch/muss.git
cd muss/
pip install -e .  # Install package
python -m spacy download en_core_web_md 

OUTPUT_DIR="llm_simplification_results"
DATA_DIR="resources/data"

mkdir -p ${OUTPUT_DIR}/muss

split="test"
for model_name in muss_en_wikilarge_mined muss_en_mined; do
	for dataset in med-easi asset news-manual-all; do
		python3 -m scripts.run_muss --input-file ${DATA_DIR}/${dataset}/${dataset}.${split}.jsonl --output-file ${OUTPUT_DIR}/muss/${dataset}-${split}_default_${model_name}.jsonl --model-name ${model_name}
	done;
done;