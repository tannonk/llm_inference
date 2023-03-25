#!/bin/bash
#SBATCH --time=00:01:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=1G
#SBATCH --array=0-1
#SBATCH --output=inf.par.%A.%a.out

# Author: T. Kew
# sbatch jobs/run_inference_all.sh

# WARINING: this script submits multiple jobs, which in turn submits their own jobs.
# This can lead to a large number of jobs being submitted, which can cause issues.
# If you are running this script, make sure you are aware of the number of jobs and keep an eye on the queue.

# Current Issues:
    # The processes are always put on the same GPU which is not ideal.

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

config_files=("$@")
# Initialize an empty array to store process IDs
pids=()

# submit a job for each config file
echo "Running JOB ID: $SLURM_ARRAY_TASK_ID"
echo "Running config: ${config_files[$SLURM_ARRAY_TASK_ID]}"

# Iterate over the list of files
for config_file in "${config_files[@]}"; do
    echo ""
    echo "$config_file"
    echo ""
    python -m run "$config_file" \
        --prompt_json prompts/p0.json \
        --examples data/asset/dataset/asset.valid.jsonl \
        --input_file data/asset/dataset/asset.test.jsonl \
        --n_refs 1 --few_shot_n 3 --seed 489 &
    pids+=($!)
done

# wait for all pids
for pid in "${pids[@]}"; do
   wait "$pid"
   echo "PID $pid done"
done