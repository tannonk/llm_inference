#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2
#SBATCH --mem=1GB
#SBATCH --time=00:01:00
#SBATCH --output=/data/tkew/projects/llm_ats/logs/%j.out

set -x

# hardcoded defaults
script_path="$( cd "$(dirname "$0")" >/dev/null 2>&1 || exit 1; pwd -P )"
BASE="$script_path/.."

module purge
module load anaconda3 multigpu a100

eval "$(conda shell.bash hook)"
conda activate && echo "CONDA ENV: $CONDA_DEFAULT_ENV"
conda activate llm_hf1 && echo "CONDA ENV: $CONDA_DEFAULT_ENV"

cd "${BASE}" && echo $(pwd) || exit 1

# determine number of GPUs as MP variable
NGPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' ' ' | wc -w)
echo "NGPUS: $NGPUS"

python -m torch.distributed.run --nproc_per_node $NGPUS inference.py "$@"