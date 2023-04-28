#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# WARNING this will attempt to run all the models with configs defined in the exp_configs/cluster folder!

# Hardcoded paths
MODEL_CONFIG_DIR=exp_configs/rtx
DATA_DIR=resources/data/prmpt_cntrl

# optional argument to specify which models to run (otherwise will run all models)
model_config_files=("$@")

if [ -z "$model_config_files" ]; then
    echo "No model config files specified, will run all configs in $MODEL_CONFIG_DIR"
    model_config_files=($(ls $MODEL_CONFIG_DIR/*.json))
fi

echo ""
echo "Running models: ${model_config_files[@]}"
echo ""

# monitor run time
SECONDS=0

for prompt_id in p0 p1 p2; do
    echo ""
    echo "### 1. PROMPT ID $prompt_id ###"
    for seed in 489 287 723; do
        echo ""
        echo "### 2. SEED $seed ###"
        for ((i=0; i<${#model_config_files[@]}; i++)); do
            echo ""
            echo "### 3. MODEL ${model_config_files[$i]} ###"
            for tgt_lvl in 1 2 3 4; do
                echo ""
                echo "### 4. TGT LVL $tgt_lvl ###"
                for ex_selector in sem_sim random; do
                    echo ""
                    echo "### 5. EX SELECTOR $ex_selector ###"
                    for few_shot_n in 0 1 2 3 4 5; do
                    # for few_shot_n in 0 1 2 3 4 5 6 7 8 9 10; do
                        echo ""
                        echo "### 6. FEW SHOT N $few_shot_n ###"

                        python -m run "${model_config_files[$i]}" \
                            -seed $seed \
                            --input_file "$DATA_DIR/newsela_manual_sents_version/0-${tgt_lvl}_test.tsv" \
                            --examples "$DATA_DIR/newsela_manual_sents_version/0-${tgt_lvl}_dev.tsv" \
                            --prompt_json prompts/$promt_json.json \
                            --output_dir resources/newsela_outputs \
                            --n_refs 1 --few_shot_n ${few_shot_n} --example_selector ${ex_selector}
                    done
                done
            done
        done
    done
done

ELAPSED="Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"
echo "Total runtime: $ELAPSED"