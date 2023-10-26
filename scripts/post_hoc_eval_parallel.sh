#!/usr/bin/env bash
# -*- coding: utf-8 -*-


function process_file {
    file=$1
    eval_file=${file%.*}.eval

    if [ -f $eval_file ]; then
        echo "Overwriting existing $eval_file ..."
    fi

    python -m evaluation.simplification_evaluation $file \
        --out_file $eval_file \
        --use_cuda \
        --lens_model_path "resources/LENS/checkpoints/epoch=5-step=6102.ckpt"

    echo "Done with $eval_file"
}

export -f process_file

# outputs_dir='resources/newsela_outputs'
# ls -l $outputs_dir/opt-13b-devtrain-common/*.jsonl $outputs_dir/opt-6.7b-common/*.jsonl $outputs_dir/opt-1.3b-common/*.jsonl | awk '{print $9}' | xargs -n 1 -P 4 -I {} bash -c 'process_file "$@"' _ {}

# outputs_dir='resources/outputs'
# ls -l $outputs_dir/*/*.jsonl | awk '{print $9}' | xargs -n 1 -P 4 -I {} bash -c 'process_file "$@"' _ {}

outputs_dir=$1
if [ -z "$outputs_dir" ]; then
    echo "expected outputs_dir as first argument"
    exit 1
fi

ls -l $outputs_dir/*.jsonl | awk '{print $9}' | xargs -n 1 -P 3 -I {} bash -c 'process_file "$@"' _ {}

echo "Done with all files"
