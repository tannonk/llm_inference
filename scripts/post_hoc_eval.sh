#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# outputs_dir='resources/outputs'
outputs_dir=$1

echo "Evaluating all files in $outputs_dir ..."

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

echo "Dine with all files"