## LLM Inference

Experimental repository for inference with LLMs supported by HuggingFace.

## Setup

To set up the environment, run the following commands:

```bash
# create a directory for source code
mkdir installs

# if working on slurm cluster, load relevant modules
ml multigpu anaconda3

# create a clean conda environment
conda create -n llm_hf1 -c conda-forge python=3.9 cudatoolkit-dev=11.6 -y
conda activate llm_hf1

# install transformers from source
git clone https://github.com/huggingface/transformers.git installs/transformers
cd installs/transformers
pip install -e .
cd ../..

# install bitsandbytes from source
git clone https://github.com/TimDettmers/bitsandbytes.git installs/bitsandbytes
cd installs/bitsandbytes
CUDA_VERSION=116 make cuda11x
python setup.py install
cd ../..

# install other deps
pip install -r requirements.txt

# check the install and CUDA dependencies
python -m bitsandbytes
```

## Examples

To run inference with LLMs available on HuggingFace, run `inference.py` passing the relevant arguments, e.g.:

```bash
python inference.py \
	--model_name_or_path "bigscience/bloom-560m" \
	--max_new_tokens 100 \
	--max_memory 0.65 \
	--batch_size 2 --num_beams 4 --num_return_sequences 3 \
	--examples "data/asset/dataset/valid.jsonl" \
	--input_file "data/asset/dataset/asset.test.orig.dummy" \
	--n_refs 2 \
	--few_shot_n 3 \
	--prompt_prefix "I want you to replace my complex sentence with simple sentence(s). Keep the meaning same, but make them simpler." \
	--example_separator "\n\n"
```

The argument `--input_file` should be a JSONL file containing the following dictionary-like object on each line:

```json
{
    "examples": 
        [
            "few-shot prompt 1", 
            "few-shot prompt 2", 
            "few-shot prompt 3",
            "etc."
        ], 
    "src": "src text / prompt for continuation"
}
```

## Data

To get the relevant datasets for reproducing experiments, use the script `scripts/fetch_data.sh`. 
This script downloads the raw data for publicly available datasets and writes the files to the `data/` directory.

```bash
bash scripts/fetch_data.sh
```

Once you have downloaded the raw datasets, we can prepare them for inference using the relevant `prepare_*.py` script.
For example, to prepare ASSET, run

```bash
python -m scripts.prepare_asset
```


## TODOs

- [ ] task-specific prompts
- [ ] datasets and data prep