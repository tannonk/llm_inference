#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# This script essentially just provides a wrapper around the run.py script
# It is intended to facilitate executing multiple experiments with a single command
# e.g. you can specify multiple model configs, seeds, and prompt ids

# Example call:
# nohup bash run.sh \
#   --input_file "resources/data/med-easi/med-easi.test.jsonl" \
#   --examples "resources/data/med-easi/med-easi.validation.jsonl" \
#   --model_configs "exp_configs/cluster/bloom-560m exp_configs/cluster/bloom-1b1.json exp_configs/cluster/bloom-3b.json" \
#   --seeds "489 287 723" \
#   --prompt_ids "p0 p1 p2" \
# > logs/medeasi_all.jobs 2>&1 &

# WARNING this will attempt to run all the models with configs defined in the exp_configs/cluster folder!

# Hardcoded paths
MODEL_CONFIG_DIR=exp_configs/cluster
DATA_DIR=resources/data
model_config_files=""
seeds="489 287 723"
prompt_ids="p0 p1 p2"

# parse arguments
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    --input_file)
    input_file="$2"
    shift 2
    ;;
    --examples)
    examples="$2"
    shift 2
    ;;
    --model_configs)
    model_config_files="$2"
    shift 2
    ;;
    --seeds)
    seeds="$2"
    shift 2
    ;;
    --prompt_ids)
    prompt_ids="$2"
    shift 2
    ;;
    *)
    echo "Unknown option: $1"
    exit 1
    ;;
  esac
done

# Convert the space-separated string into an array
IFS=' ' read -ra model_config_files <<< "$model_config_files"
IFS=' ' read -ra seeds <<< "$seeds"
IFS=' ' read -ra prompt_ids <<< "$prompt_ids"

# Now you can use file1, file2, and values_array in your script


if [ -z "$model_config_files" ]; then
    echo "No model config files specified, will run all configs in $MODEL_CONFIG_DIR"
    model_config_files=($(ls $MODEL_CONFIG_DIR/*.json | grep -v "openai-*")) # ignore openai configs!
fi

echo ""
echo "Input File: $input_file"
echo "Examples: $examples"
echo "Seeds: ${seeds[@]}"
echo "Prompt IDs: ${prompt_ids[@]}"
echo "Models: ${model_config_files[@]}"
echo ""

# iterate through array using a counter
for prompt_id in "${prompt_ids[@]}"; do
    echo ""
        echo "### PROMPT ID $prompt_id ###"
    echo ""
    for seed in "${seeds[@]}"; do
        echo ""
        echo "### SEED $seed ###"
        echo ""
        for model_config_file in "${model_config_files[@]}"; do
            echo ""
            echo "### $model_config_file ###"

            python -m run "$model_config_file" \
                --seed $seed \
                --input_file $input_file \
                --examples $examples \
                --prompt_json prompts/$prompt_id.json \
                --n_refs 1 --few_shot_n 3
        done
    done
done