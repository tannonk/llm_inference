## LLM Inference

Experimental repository for text simplification with LLMs.

## Setup

To set up the environment, run the following commands:

```bash
# create a directory for source code
mkdir installs

# create a directory for data, models, outputs, etc.
mkdir -p resources/data resources/outputs resources/models
# or alternatively, create a symlink
ln -s <path_to_storage> resources

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

# install other deps
pip install -r requirements.txt

# check the install and CUDA dependencies
python -m bitsandbytes

# For evaluation purposes, we also require the following packages
git clone https://github.com/feralvam/easse.git installs/easse
cd installs/easse
pip install -e .
cd ../..

git clone https://github.com/Yao-Dou/LENS.git installs/LENS
cd LENS/lens
pip install -e .
cd ../../..

```

**Note:** `bitsandbytes` is only required for running inference with 8bit quantization. If you have any problems installing this library and don't intend on running inference locally, you can skip this dependency.

## Models

At the moment, we support the following models:

- Bloom / Bloomz
- Llama
- OPT (up to 66B) / OPT-IML
- GPT-J / GPT-NeoX
- T5 / T0 / Flan-T5
- UL2 / Flan-UL2
- OpenAI models (via API)
- Cohere models (via API)

## Data

To get the relevant datasets for reproducing experiments, use the script [`scripts/fetch_datasets.sh`](./scripts/fetch_datasets.sh). 
This script downloads the raw data for publicly available datasets and writes the files to the `resources/data/` directory.

```bash
bash scripts/fetch_datasets.sh
```

Once you have downloaded the raw datasets, we can prepare them for inference using the relevant `prepare_*.py` script.
For example, to prepare ASSET, run

```bash
python -m scripts.prepare_asset
```

See the [readme](./scripts/README.md) for more details on the format.

## How to run

To facilitate running inference with different datasets, models, random seeds and prompts, use [./run.sh](./run.sh).

It accepts the following arguments:
- `--input_file` (test set)
- `--examples` (validation set from which we sample few-shot examples) 
- `--model_configs` a space-delimited list of exp_configs (i.e. models to run, see [./exp_configs/](./exp_configs/)).
- `--seeds` a space-delimited list of seeds to use (for consistency, use 489 287 723)
- `--prompt_ids` a space-delimited list of predefined prompts (see [./prompts/](./prompts/))

For example to run inference on Med-EASI with 3 models, 2 seeds and 1 prompt would like this:

```bash
nohup bash run.sh \
 --input_file "resources/data/med-easi/med-easi.test.jsonl" \
 --examples "resources/data/med-easi/med-easi.validation.jsonl" \
 --model_configs "exp_configs/cluster/bloom-560m.json exp_configs/cluster/bloom-1b1.json exp_configs/cluster/bloom-3b.json" \
 --seeds "489 723" \
 --prompt_ids "p1" logs/medeasi.jobs 2>&1 &
```

Similarly, for API models, running inference with OpenAI models can be done as follows:

```bash
nohup bash run.sh \
 --input_file "resources/data/med-easi/med-easi.test.jsonl" \
 --examples "resources/data/med-easi/med-easi.validation.jsonl" \
 --model_configs "exp_configs/rtx/openai-gpt-3.5-turbo.json exp_configs/rtx/openai-text-babbage-001.json exp_configs/rtx/openai-text-davinci-002.json exp_configs/rtx/openai-text-ada-001.json exp_configs/rtx/openai-text-curie-001.json exp_configs/rtx/openai-text-davinci-003.json" \
 --seeds "287 489 723" \
 --prompt_ids "p0 p1 p2" logs/all_medeasi_openai.jobs 2>&1 &
```

**Note:** evaluation runs immediately after inference so it pays to do this on a GPU server to efficiently compute mode-based metrics!

### Wrappers on wrappers

Things got a bit out-of-hand due to the number of experiments to run and in the hope of supporting different hardware setups.
The heirarchy of scripts is currently:

```bash
./run.sh
    └── ./run.py
            ├── ./slurm_scripts/run_inference_on_*.sh
            │       └── ./inference.py
            └──./slurm_scripts/run_evaluation.sh
                    └── ./evaluation/simplification_evaluation.py
```	

Below we provide more information on the lower-level scripts.

#### run.py

[`./run.py`](./run.py) executes a single **inference** run followed by **evaluation** of model outputs.

For example, to run inference on `ASSET` with `bloom-560m` with prompt `p0`, you can run:

```bash
python -m run \
    --use_slurm True \
    --ntasks 1 \
    --cpus_per_task 1 \
    --gres gpu:T4:1 \
    --mem 20GB \
    --time 00:30:00 \
    --batch_size 8 \
    --seed 489 \
    --model_name_or_path bigscience/bloom-560m \
    --examples resources/data/asset/dataset/asset.valid.jsonl \
    --input_file resources/data/asset/dataset/asset.test.jsonl \
    --prompt_json prompts/p0.json \
    --n_refs 1 --few_shot_n 3 \
    --dry_run False # set to True to inspect the command calls without actually executing anything
```

Alternatively, you can also pass a json file from [./exp_configs/](./exp_configs/) in position 1 with some or all of the arguments predefined.

For example, on a server with RTX 3090 (24GB) GPUs, you could use the following:

```bash
python -m run exp_configs/rtx/bloom-560m.json \
    --seed 489 \
    --examples resources/data/asset/dataset/asset.valid.jsonl \
    --input_file resources/data/asset/dataset/asset.test.jsonl \
    --prompt_json prompts/p0.json \
    --n_refs 1 --few_shot_n 3 \
    --dry_run False # set to True to inspect the command calls without actually executing anything
```

This scripts produces the following files:

- `<output_file>.jsonl`: The predictions of the model on the input file.
- `<output_file>.json`: Command line arguments used for the inference run.
- `<output_file>.log`: Log file of the inference run.
- `<output_file>.eval`: Log file of the automatic evaluation with results. 

#### inference.py

[./inference.py](./inference.py) is the script that performs inference with a LLM.

For example, to use `bigscience/bloom-1b1` for local inference, you could execute the following:

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
    --input_file "resources/data/asset/dataset/asset.test.jsonl" \
    --examples "resources/data/asset/dataset/asset.valid.jsonl" \
    --n_refs 1 \
    --few_shot_n 3 \
    --output_dir "resources/outputs" \
    --prompt_json "prompts/p0.json"
```

where:
- `--input_file` is a either a .txt file, with one input sentence per line OR a JSONL file produced by `scripts/prepare_*.py`. For consistency we recommend the latter.
- `--examples` is a JSONL file produced by `scripts/prepare_*.py`, containing validation set examples that may be selected as few-shot examples.
- NB I: by default, an additional JSON file will be generated which persists the inference parameters used for generation.
- NB II: specify `--output_dir ''` to print your outputs to stdout (good for debugging/development purposes)

The same script can be used for inference with closed-source models behind APIs.

For example:

```bash
python -m inference \
    --model_name_or_path cohere-command-xlarge-nightly \
    --input_file "resources/data/asset/dataset/asset.test.jsonl" \
    --examples "resources/data/asset/dataset/asset.valid.jsonl" \
    --n_refs 1 \
    --few_shot_n 3 \
    --output_dir "resources/outputs" \
    --prompt_json "prompts/p0.json"
```

**Note:** for this to work you will have to add your keys to [`./api_secrets.py`](./api_secrets.py) so that `COHERE_API_KEY` and `OPENAI_API_KEY` are exposed to the library.

Be aware that API models cost money! 

The table below provides the approximate cost of running the OpenAI models on the ASSET test set (359 examples) with a **single** seed/prompt (using 3 Few-shot examples with the pre-defined prompts). **Note:** that the prompts differ in length and therefore overall cost.

|          Model             |  `p0`  |  `p1`  |  `p2`  |        Pricing        |
| -------------------------- | ------ | ------ | ------ | --------------------- |
| openai-gpt-3.5-turbo       | $0.172 | $0.189 | $0.22  | $0.002 / 1k tokens    |
| openai-text-ada-001        | $0.035 | $0.038 | $0.046 | $0.0004 / 1k tokens   |
| openai-text-babbage-001    | $0.044 | $0.047 | $0.057 | $0.0005 / 1k tokens   |
| openai-text-curie-001      | $0.175 | $0.19  | $0.23  | $0.002 / 1k tokens    |
| openai-text-davinci-002    | $1.75  | $1.85  | $2.26  | $0.02 / 1k tokens     |
| openai-text-davinci-003    | $1.75  | $1.85  | $2.26  | $0.02 / 1k tokens     |
| approx. # TOKENS processed |  ~86K  |  ~93K  |  ~113k |                       |

For each prompt, we run inference with 3 seeds, totalling 9 inference runs per dataset. 
Therefore the maximal cost of running all experiments with OpenAI most expensive models is approximately $54 (2x3x3x3).

#### evaluation.simplification_evaluation.py

[`./evaluation/simplification_evaluation.py`](./evaluation/simplification_evaluation.py) computes automatic metrics using EASSE and other libraries.

We recommend running evaluation with a GPU in order to compute model-based metrics (e.g. LENS, BERTScore, PPL).

#### Reference-free Evaluation Metrics

We also compute all automatic metrics on the ground truth simplificiations to provide a reference point for reference-free metrics such as FKGL and QE statistics.

To prepare the ground truth texts as model outputs and evaluate, run:

```bash
python scripts/prepare_ground_truth_as_outputs.py \
    resources/data/asset/dataset/asset.test.jsonl \
    resources/outputs/ground_truths/asset.test.jsonl

python -m evaluation.simplification_evaluation \
    resources/outputs/ground_truths/asset.test.jsonl \
    --out_file resources/outputs/ground_truths/asset.test.eval \
    --use_cuda
```

### About the prompts

To construct prompts flexibly, we use [LangChain](https://github.com/hwchase17/langchain).

A valid prompt may look something like the following:

```
I want you to replace my complex sentence with simple sentence(s). Keep the meaning same, but make them simpler.

Complex: The Hubble Space Telescope observed Fortuna in 1993.
Simple: 0: The Hubble Space Telescope spotted Fortuna in 1993.

Complex: Order # 56 / CMLN of 20 October 1973 prescribed the coat of arms of the Republic of Mali.
Simple: 0: In 1973, order #56/CMLN described the coat of arms for the Republic of Mali.

Complex: One side of the armed conflicts is composed mainly of the Sudanese military and the Janjaweed, a Sudanese militia group recruited mostly from the Afro-Arab Abbala tribes of the northern Rizeigat region in Sudan.
Simple:
```

This example corresponds to the T1 prompt described in [Feng et al., 2023](http://arxiv.org/abs/2302.11957).

Prompts can be defined on-the-fly at inference time by passing the relevant arguments. To do this for the example prompt above, pass the following arguments:

```bash
--prompt_prefix "I want you to replace my complex sentence with simple sentence(s). Keep the meaning same, but make them simpler."
--promt_suffix "Complex: {input}\nSimple:"
--prompt_template "Complex: {complex}\nSimple: {simple}"
--example_separator "\n\n"
--prompt_format "prefix_initial"
```

However, for reproducibility, we recommend using pre-defined prompts. These contain these relevant fields and easily be used for inference by passing them with the `--prompt_json` argument.
The directory [prompts](./prompts) contains a set of pre-defined prompts in JSON format. See the [readme](./prompts/README.md) for more details.


## Results

The generated outputs and automtic evaluation results can be found [here](https://github.com/tannonk/llm_simplification_results).
**Note:** this is a private repo for now. Please contact Tannon Kew (kew@cl.uzh.ch) to get access.

**Note:** for ASSET, which has multiple references, we randomly select 1 as the stand-in model output text and use all the others as references.

## Limitations & Known Issues

- LLMs don't know when to stop. Thus, they typically generate sequences up to the specified `max_new_tokens`. The function `postprocess_model_outputs()` is used to extract the single relevant model output from a long generation sequence and is currently pretty rough.
- Setting `--n_refs` > 1 allows for a few-shot prompt example to have multiple possible targets (e.g. sampled from multiple validation set reference sentences). The current method of handling these is to enumerate them starting at 0, but this doesn't seem very elegant or intuitive.

## TODOs

You can find a list of pending experiments in this [checklist](https://github.com/tannonk/llm_inference/blob/main/reports/checklist/checklist.md). Feel free to suggest any new setting, model or dataset.