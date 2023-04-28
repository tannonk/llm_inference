#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# Example call:
# bash run_openai.sh openai-text-ada-001

# WARNING this will attempt to run all the models with configs defined in the exp_configs/cluster folder!
MODEL_CONFIG_DIR=exp_configs/cluster
DATA_DIR="resources/data"

# optional argument to specify which models to run (otherwise will run all models)
model_config_files=("$@")

if [ -z "$model_config_files" ]; then
    echo "No model config files specified, will run all configs in $MODEL_CONFIG_DIR"
    model_config_files=($(ls $MODEL_CONFIG_DIR/openai-*.json))
fi

echo ""
echo "Running models: ${model_config_files[@]}"
echo ""

# iterate through array using a counter
for prompt_id in p0 p1 p2; do
    echo ""
        echo "### PROMPT ID $prompt_id ###"
    echo ""
    for seed in 489 287 723; do
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
                --prompt_json prompts/$prompt_id.json \
                --n_refs 1 --few_shot_n 3
        done
    done
done