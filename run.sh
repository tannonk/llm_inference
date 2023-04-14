#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# WARNING this will attempt to run all the models with configs defined in the exp_configs/cluster folder!

MODEL_CONFIG_DIR=exp_configs/cluster

# optional argument to specify which models to run
model_config_files=("$@")

if [ -z "$model_config_files" ]; then
    echo "No model config files specified, will run all configs in $MODEL_CONFIG_DIR"
    model_config_files=($(ls $MODEL_CONFIG_DIR/*.json))
fi

echo ""
echo "Running models: ${model_config_files[@]}"
echo ""

# iterate through array using a counter
for ((i=0; i<${#model_config_files[@]}; i++)); do
    echo ""
    echo "### $i ${model_config_files[$i]} ###"

    python -m run "${model_config_files[$i]}" \
        --seed 489 \
        --examples resources/data/asset/dataset/asset.valid.jsonl \
        --input_file resources/data/asset/dataset/asset.test.jsonl \
        --prompt_json prompts/p0.json \
        --n_refs 1 --few_shot_n 3

done
