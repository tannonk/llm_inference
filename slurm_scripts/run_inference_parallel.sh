#!/bin/bash
#SBATCH --time=00:01:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=1G
#SBATCH --array=0-1
#SBATCH --output=inf.par.%A.%a.out
#SBATCH --partition=lowprio

# Author: T. Kew
# sbatch jobs/run_inference_parallel.sh

# hardcoded defaults
BASE="/data/tkew/projects/llm_ats/" # expected path on slurm cluster
if [ ! -d "$BASE" ]; then
    echo "Failed to locate BASE directory '$BASE'. Inferring BASE from script path..."
    SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
    BASE="$(dirname "$SCRIPT_DIR")"
fi

module purge
module load anaconda3 multigpu a100

eval "$(conda shell.bash hook)"
conda activate && echo "CONDA ENV: $CONDA_DEFAULT_ENV"
conda activate llm_hf1 && echo "CONDA ENV: $CONDA_DEFAULT_ENV"

cd "${BASE}" && echo $(pwd) || exit 1

# config_dir="exp_configs"
# config_files=($(ls $config_dir))

# config_files=("exp_configs/llama-7B-3-1-asset-p0.json" "exp_configs/bloom-560m-3-1-asset-p0.json")
config_files=("$@")

# submit a job for each config file
echo "Running JOB ID: $SLURM_ARRAY_TASK_ID"
echo "Running config: ${config_files[$SLURM_ARRAY_TASK_ID]}"

srun python -m slurm_scripts.submit_inference "${config_files[$SLURM_ARRAY_TASK_ID]}"

# Iterate over the list of files
# for config_file in "${config_files[@]}"; do
#     # Do something with each file
#     echo "$config_file"
# done

# for exp_config in "$config_dir"/*.json; do
#     echo "Found config: $exp_config"

# exp_ids=("baseline" "qu_ctxt_aug1" "qu_ctxt_aug5" "short_qu_ctxt_aug5" "pos_sent_ctxt_aug5" "neg_sent_ctxt_aug5" "ambig_qu_ctxt_aug5" "ambig_excl_ctxt_aug5" "excl_ctxt_aug5" "hedging_contrast_ctxt_aug5" "hedging_management_ctxt_aug5" "hedging_evasion_ctxt_aug5" "e_words_ctxt_aug5" "d_words_ctxt_aug5" "i_words_ctxt_aug5" "n_words_ctxt_aug5")

# # launches a single experiment job for each exp_id in parallel
# srun python generation_exp.py \
#     --model_dir "$model_path" \
#     --output_dir "$output_dir" \
#     --dataset "$dataset" \
#     --batch_size "$batch_size" \
#     --exp_id "${exp_ids[$SLURM_ARRAY_TASK_ID]}"

echo ""
echo "Done."
echo ""