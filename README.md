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
# NOTE: we recommend using CUDA 11.6, but others may work too
conda create -n llm_hf1 -c conda-forge python=3.9 cudatoolkit=11.6 cudatoolkit-dev=11.6 -y
conda activate llm_hf1

# install transformers from source
git clone https://github.com/huggingface/transformers.git installs/transformers
cd installs/transformers
pip install -e .
cd ../..

# NOTE: for efficient inference, we use 8bit quantization with bitsandbytes. 
# This requires Turing or Ampere GPUs (RTX 20s, RTX 30s, A40-A100, T4+)
# install bitsandbytes from source
git clone https://github.com/TimDettmers/bitsandbytes.git installs/bitsandbytes
cd installs/bitsandbytes
CUDA_VERSION=116 make cuda11x
python setup.py install
cd ../..

# # install promptsource from source
# git clone https://github.com/bigscience-workshop/promptsource.git installs/promptsource
# cd installs/promptsource
# pip install -e .

# install other deps
pip install -r requirements.txt

# check the install and CUDA dependencies
python -m bitsandbytes
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

## Examples

To run inference with LLMs available on HuggingFace, run `inference.py` passing the relevant arguments, e.g.:

```bash
python -m inference \
	--model_name_or_path "bigscience/bloom-1b1" \
	--max_new_tokens 100 \
	--max_memory 0.65 \
	--batch_size 8 \
	--num_beams 1 \
	--num_return_sequences 1 \
	--do_sample True \
	--top_p 0.9 \
	--input_file "data/asset/dataset/asset.test.jsonl" \
	--examples "data/asset/dataset/asset.valid.jsonl" \
	--n_refs 1 \
	--few_shot_n 3 \
	--prompt_prefix "I want you to replace my complex sentence with simple sentence(s). Keep the meaning same, but make them simpler." \
	--output_dir "data/outputs" \
```

where:
- `--input_file` is a either a .txt file, with one input sentence per line or a JSONL file produced by `scripts.prepare_*.py`. For consistency we recommend the latter.
- `--examples` is a JSONL file produced by `scripts.prepare_*.py`, containing validation set examples that may be selected as few-shot examples.
- `--prompt_prefix` is a string prefix used by LangChain to construct the prompt.
- NB I: by default, an additional JSON file will be generated which persists the inference parameters used for generation.
- NB II: specify `--output_dir ''` to print your outputs to stdout (good for debugging/development purposes)

#### For Slurm Users Only

For experiments executed on a slurm cluster, we provide relevant scripts in `slurm_scripts`. Here you can launch an inference job by specfying the SBATCH commands and inference paramters using `slurm_scripts/submit_inference.py`, e.g.:

```bash
python -m slurm_scripts.submit_inference \
	--use_slurm True --ntasks 1 --cpus_per_task 1 --gres gpu:T4:1 --mem 32GB --time 01:00:00 \ `# SBATCH commands`
	--model_name_or_path "facebook/opt-iml-1.3b" \ `# inference commands`
	--input_file "data/asset/dataset/asset.test.jsonl" \
	--examples "data/asset/dataset/asset.valid.jsonl" \
	--output_dir "data/outputs" \
	--prompt_prefix "I want you to replace my complex sentence with simple sentence(s). Keep the meaning same, but make them simpler."
```

This script will also write a job log file in the `--output_dir`.

## Prompting

To construct prompts flexibly, we use [LangChain](https://github.com/hwchase17/langchain).

A valid prompt may look something like the following:

```
I want you to replace my complex sentence with simple sentence(s). Keep the meaning same, but make them simpler.

Complex: The Hubble Space Telescope observed Fortuna in 1993.
Simple: 0: The Hubble Space Telescope spotted Fortuna in 1993.

Complex: Order # 56 / CMLN of 20 October 1973 prescribed the coat of arms of the Republic of Mali.
Simple: 0: In 1973, order #56/CMLN described the coat of arms for the Republic of Mali.

Complex: In the 1950s Camus devoted his efforts to human rights.
Simple: 0: In the 1950s Camus worked on human rights.

Complex: One side of the armed conflicts is composed mainly of the Sudanese military and the Janjaweed, a Sudanese militia group recruited mostly from the Afro-Arab Abbala tribes of the northern Rizeigat region in Sudan.
Simple:
```

The example corresponds to the T1 prompt described in [Feng et al., 2023](http://arxiv.org/abs/2302.11957).

## Observations

Below are some observations from running inference with LLMs:

1. Beam search uses significantly more GPU memory compared to sampling-based decoding methods with `num_beams=1`
<!-- 2. Inference with `bigscience/bloom-560m` on a single T4 (16GB) GPU takes ~10 seconds per batch with `batch_size=4, max_new_tokens=100, num_beams=1, num_return_sequences=1, do_sample=True, top_p=0.9` and uses ~6GB GPU memory -->
<!-- 3. Inference with `bigscience/bloom-1b1` on a single T4 (16GB) GPU takes ~10 seconds per batch with `batch_size=4, max_new_tokens=100, num_beams=1, num_return_sequences=1, do_sample=True, top_p=0.9` and uses ~8GB GPU memory -->

The following table is based off of generating with the following params (unless otherwise specified): `batch_size=4, max_new_tokens=100, num_beams=1, num_return_sequences=1, do_sample=True, top_p=0.9`

| 		Model 		| 	Footprint       | Loading time  | Inference time  | Inference GPU mem |  # GPUS  |
| :---------------: | :---------------: | :-----------: | :-------------: | :---------------: | :------: |
| bigscience/bloom-560m | 0.78GB        |        ?      | ~10 secs (bs=4) |         6GB       |  1 (T4)  |
| bigscience/bloom-1b1  | 1.35GB        |        ?      | ~10 ses (bs=4)  |         8GB       |  1 (T4)  |
| bigscience/bloom  | 	167.5 GB	    |    15 mins    | ~45 secs (bs=8) |          ?        | 4 (A100) |
| facebook/opt-iml-max-30b |  28.26GB   |     5 mins    | ~10 secs (bs=8) |          ?        | 1 (A100) |
| facebook/opt-66b      |    28.26GB    |     5 mins    | ~10 secs (bs=8) |          ?        | 1 (A100) |


## Limitations

LLMs don't know when to stop. Thus, they typically generate sequences up to the specified `max_new_tokens`. 
The function `postprocess_model_outputs()` is used to extract the single relevant model output from a long generation sequence and is currently pretty rough.

## TODOs

- [x] task-specific prompts
- [ ] datasets and data prep
	- [ ] Newsela
	- [ ] Hsplit
