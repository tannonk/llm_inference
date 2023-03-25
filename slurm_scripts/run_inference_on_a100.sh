#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:A100:4
#SBATCH --mem=100GB
#SBATCH --time=02:00:00

# __Author__: Tannon Kew (kew@cl.uzh.ch)

# set -x

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

python inference.py "$@"