#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# outputs_dir='resources/outputs'
outputs_dir=$1

if [ -z "$outputs_dir" ]; then
    echo "expected outputs_dir as first argument"
    exit 1
fi

for file in $(ls -l $outputs_dir/*.jsonl | awk '{print $9}'); do
    # echo $file
    eval_file=${file%.*}.eval
    # echo $eval_file
    if [ -f $eval_file ]; then
        echo "Overwriting existing $eval_file ..."
    fi

    python -m evaluation.simplification_evaluation $file \
        --out_file $eval_file \
        --use_cuda \
        --lens_model_path "resources/LENS/checkpoints/epoch=5-step=6102.ckpt"

    echo "Done with $eval_file"
done

# outputs_dir='resources/newsela_outputs'

# for file in $(ls -l $outputs_dir/*-common/*.jsonl | awk '{print $9}'); do
#     eval_file=${file%.*}.eval
    
#     if [ -f $eval_file ]; then
#         echo "Overwriting existing $eval_file ..."
#     fi

#     python -m evaluation.simplification_evaluation $file \
#         --out_file $eval_file \
#         --use_cuda \
#         --lens_model_path "resources/LENS/checkpoints/epoch=5-step=6102.ckpt"

#     echo "Done with $eval_file"
# done

# echo "Done with all files"