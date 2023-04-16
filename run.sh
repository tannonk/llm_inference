#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# WARNING this will attempt to run all the models with configs defined in the exp_configs/cluster folder!

# Hardcoded paths
MODEL_CONFIG_DIR=exp_configs/cluster
DATA_DIR=resources/data

# optional argument to specify which models to run (otherwise will run all models)
model_config_files=("$@")

if [ -z "$model_config_files" ]; then
    echo "No model config files specified, will run all configs in $MODEL_CONFIG_DIR"
    model_config_files=($(ls $MODEL_CONFIG_DIR/*.json))
fi

echo ""
echo "Running models: ${model_config_files[@]}"
echo ""

# iterate through array using a counter
# for seed in 489 287 723; do
for seed in 287 723; do
    echo ""
    echo "### SEED $seed ###"
    echo ""
    for ((i=0; i<${#model_config_files[@]}; i++)); do
        echo ""
        echo "### $i ${model_config_files[$i]} ###"

        python -m run "${model_config_files[$i]}" \
            --seed $seed \
            --input_file $DATA_DIR/asset/dataset/asset.test.jsonl \
            --examples $DATA_DIR/asset/dataset/asset.valid.jsonl \
            --prompt_json prompts/p0.json \
            --n_refs 1 --few_shot_n 3
    done
done
