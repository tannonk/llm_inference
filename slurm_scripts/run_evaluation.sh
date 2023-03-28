#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --time=01:00:00
#SBATCH --partition=lowprio

# __Author__: Tannon Kew (kew@cl.uzh.ch)

# set -x

# hardcoded defaults
BASE="/data/tkew/projects/llm_ats/" # expected path on slurm cluster
if [ ! -d "$BASE" ]; then
    echo "Failed to locate BASE directory '$BASE'. Inferring BASE from script path..."
    SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
    BASE="$(dirname "$SCRIPT_DIR")"
fi

INFILE="$1"
OUTFILE="$2"

module purge
module load anaconda3 gpu

eval "$(conda shell.bash hook)"
conda activate && echo "CONDA ENV: $CONDA_DEFAULT_ENV"
conda activate llm_hf1 && echo "CONDA ENV: $CONDA_DEFAULT_ENV"

cd "$BASE" && echo $(pwd) || exit 1

python -m evaluation.simplification_evaluation "$INFILE" --out_file "$OUTFILE" --use_cuda