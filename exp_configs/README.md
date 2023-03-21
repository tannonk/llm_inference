This directory contains predefined experiment settings for various models. The purpose of these config file is to facilitate inference run experiments, specifically on a slurm cluster.

We specify model-specific arguments such as GPU memory requirements in the appropriate JSON file, which can be passes to `slurm_scripts.submit_inference` as a positional argument in position 1.

We then specify experiment specific arguments as usual, e.g.:

```bash

python -m slurm_scripts.submit_inference exp_configs/bloom-560m-3-1-asset-p0.json --prompt_json "prompts/p0.json" --examples "data/asset/dataset/asset.valid.jsonl" --input_file "data/asset/dataset/asset.test.orig"

```