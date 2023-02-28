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
	--model_name_or_path "bigscience/bloom-1b1" \
	--max_new_tokens 100 \
	--max_memory 0.65 \
	--batch_size 8 \
	--num_beams 1 \
	--num_return_sequences 1 \
	--do_sample True \
	--top_p 0.9 \
	--input_file "data/asset/dataset/asset.test.orig" \
	--examples "data/asset/dataset/valid.jsonl" \
	--n_refs 1 \
	--few_shot_n 3 \
	--prompt_prefix "I want you to replace my complex sentence with simple sentence(s). Keep the meaning same, but make them simpler." \
	--output_file  "data/outputs/bloom-1b1/asset.test"
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

## Observations

Below are some observations from running inference with LLMs:

1. Beam search uses significantly more GPU memory compared to sampling-based decoding methods with `num_beams=1`
2. Inference with `bigscience/bloom-560m` on a single T4 (16GB) GPU takes ~10 seconds per batch with `batch_size=4, max_new_tokens=100, num_beams=1, num_return_sequences=1, do_sample=True, top_p=0.9` and uses ~6GB GPU memory
3. Inference with `bigscience/bloom-1b1` on a single T4 (16GB) GPU takes ~10 seconds per batch with `batch_size=4, max_new_tokens=100, num_beams=1, num_return_sequences=1, do_sample=True, top_p=0.9` and uses ~8GB GPU memory
4. The model footprint for `bigscience/bloom-560m` with `load_in_8bit=True` is ~0.78GB
5. The model footprint for `bigscience/bloom-1b1` with `load_in_8bit=True` is ~1.35GB


## Limitations

LLMs don't know when to stop. Thus, they typically generate sequences up to the specified `max_new_tokens`. 
The function `postprocess_model_outputs()` is used to extract the single relevant model output from a long generation sequence and is currently pretty rough.

## TODOs

- [x] task-specific prompts
- [ ] datasets and data prep