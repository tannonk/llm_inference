#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# WARNING this will attempt to run all the models with configs defined in the exp_configs/cluster folder!
#
# Example call:
# nohup bash prmpt_cntrl/run.sh exp_configs/cluster/opt-1.3b.json > logs/prmpt_cntrl_opt1.3b.jobs 2>&1 &

# Hardcoded paths
# hardcoded defaults
BASE="/data/tkew/projects/llm_ats/" # expected path on slurm cluster
if [ ! -d "$BASE" ]; then
    echo "Failed to locate BASE directory '$BASE'. Inferring BASE from script path..."
    SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
    BASE="$(dirname "$SCRIPT_DIR")"
fi

module purge
# module load anaconda3 multigpu a100 # use for a100s
module load anaconda3 gpu

eval "$(conda shell.bash hook)"
conda activate && echo "CONDA ENV: $CONDA_DEFAULT_ENV"
conda activate llm_hf1 && echo "CONDA ENV: $CONDA_DEFAULT_ENV"

cd "${BASE}" && echo $(pwd) || exit 1

MODEL_CONFIG_DIR=$BASE/exp_configs/cluster
DATA_DIR=$BASE/resources/data/prmpt_cntrl

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

for ((i=0; i<${#model_config_files[@]}; i++)); do
    echo ""
    echo "### 1. MODEL ${model_config_files[$i]} ###"
    # for prompt_id in p0 p1 p2; do
    for prompt_id in p0 ; do
        echo ""
        echo "### 2. PROMPT ID $prompt_id ###"
        for seed in 489 287 723; do
            echo ""
            echo "### 3. SEED $seed ###"
            for tgt_lvl in 1 2 3 4; do
                echo ""
                echo "### 4. TGT LVL $tgt_lvl ###"
                for ex_selector in sem_sim random; do
                    echo ""
                    echo "### 5. EX SELECTOR $ex_selector ###"
                    # for few_shot_n in 0 1 2 3 4 5; do
                    for few_shot_n in 0 1 2 3 4 5 6 7 8 9 10; do
                        echo ""
                        echo "### 6. FEW SHOT N $few_shot_n ###"

                        python -m run "${model_config_files[$i]}" \
                            --seed ${seed} \
                            --input_file "$DATA_DIR/newsela_manual_sents_version/0-${tgt_lvl}_test.tsv" \
                            --examples "$DATA_DIR/newsela_manual_sents_version/0-${tgt_lvl}_dev.tsv" \
                            --prompt_json "$BASE/prompts/$prompt_id.json" \
                            --output_dir "$BASE/resources/newsela_outputs" \
                            --n_refs 1 --few_shot_n ${few_shot_n} --example_selector ${ex_selector}
                        
                        # disctractor few-shot examples
                        case $tgt_lvl in
                            1)
                                echo "Distractors for TARGET LEVEL $tgt_lvl: $DATA_DIR/newsela_manual_sents_version/0-234_dev.tsv"

                                python -m run "${model_config_files[$i]}" \
                                    --seed ${seed} \
                                    --input_file "$DATA_DIR/newsela_manual_sents_version/0-${tgt_lvl}_test.tsv" \
                                    --examples "$DATA_DIR/newsela_manual_sents_version/0-234_dev.tsv" \
                                    --prompt_json "$BASE/prompts/$prompt_id.json" \
                                    --output_dir "$BASE/resources/newsela_outputs" \
                                    --n_refs 1 --few_shot_n ${few_shot_n} --example_selector ${ex_selector}
                                
                                ;;
                            2)
                                echo "Distractors for TARGET LEVEL $tgt_lvl: $DATA_DIR/newsela_manual_sents_version/0-134_dev.tsv"
                                
                                python -m run "${model_config_files[$i]}" \
                                    --seed ${seed} \
                                    --input_file "$DATA_DIR/newsela_manual_sents_version/0-${tgt_lvl}_test.tsv" \
                                    --examples "$DATA_DIR/newsela_manual_sents_version/0-134_dev.tsv" \
                                    --prompt_json "$BASE/prompts/$prompt_id.json" \
                                    --output_dir "$BASE/resources/newsela_outputs" \
                                    --n_refs 1 --few_shot_n ${few_shot_n} --example_selector ${ex_selector}
                                
                                ;;
                            3)
                                echo "Distractors for TARGET LEVEL $tgt_lvl: $DATA_DIR/newsela_manual_sents_version/0-124_dev.tsv"
                                
                                python -m run "${model_config_files[$i]}" \
                                    --seed ${seed} \
                                    --input_file "$DATA_DIR/newsela_manual_sents_version/0-${tgt_lvl}_test.tsv" \
                                    --examples "$DATA_DIR/newsela_manual_sents_version/0-124_dev.tsv" \
                                    --prompt_json "$BASE/prompts/$prompt_id.json" \
                                    --output_dir "$BASE/resources/newsela_outputs" \
                                    --n_refs 1 --few_shot_n ${few_shot_n} --example_selector ${ex_selector}
                                
                                ;;
                            4)
                                echo "Distractors for TARGET LEVEL $tgt_lvl: $DATA_DIR/newsela_manual_sents_version/0-123_dev.tsv"
                                
                                python -m run "${model_config_files[$i]}" \
                                    --seed ${seed} \
                                    --input_file "$DATA_DIR/newsela_manual_sents_version/0-${tgt_lvl}_test.tsv" \
                                    --examples "$DATA_DIR/newsela_manual_sents_version/0-123_dev.tsv" \
                                    --prompt_json "$BASE/prompts/$prompt_id.json" \
                                    --output_dir "$BASE/resources/newsela_outputs" \
                                    --n_refs 1 --few_shot_n ${few_shot_n} --example_selector ${ex_selector}
                                
                                ;;
                        esac
                    done
                done
            done
        done
    done
done

ELAPSED="Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"
echo "Total runtime: $ELAPSED"