#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# __Author__ = 'Sweta Agrawal'
# __Email__ = 'sweagraw@umd.edu'
# __Date__ = '2023-05-26'


OUTPUT_DIR="resources/outputs"
DATA_DIR="resources/data"

mkdir -p ${OUTPUT_DIR}/muss

split="test"
for model_name in muss_en_wikilarge_mined muss_en_mined; do
	for dataset in med-easi asset news-manual-all; do
		python3 -m scripts.run_muss --input-file ${DATA_DIR}/${dataset}/${dataset}.${split}.jsonl --output-file ${OUTPUT_DIR}/muss/${dataset}-${split}_default_${model_name}.jsonl --model-name ${model_name}
	done;
done;