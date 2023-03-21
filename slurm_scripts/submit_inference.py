#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __Author__ = 'Tannon Kew'
# __Email__ = 'kew@cl.uzh.ch
# __Date__ = '2023-03-09'

"""

This is a wrapper to facilitate the execution of inference jobs on a slurm cluster

Example Call:

python -m slurm_scripts.submit_inference \
    --use_slurm True \
    --ntasks 1 \
    --cpus_per_task 1 \
    --gres gpu:T4:1 \
    --mem 32GB \
    --time 00:30:00 \
    --batch_size 8 \
    --seed 489 \
    --model_name_or_path "bigscience/bloom-560m" \
    --examples "data/asset/dataset/asset.valid.jsonl" \
    --input_file "data/asset/dataset/asset.test.jsonl" \
    --prompt_json "p0.json"

alternatively, you can pass a json file with all the arguments:

python -m slurm_scripts.submit_inference exp_configs/bloom-560m-3-1.json

"""


import os, sys
import subprocess
import json

from dataclasses import dataclass, field

from transformers import HfArgumentParser
from llm_inference import InferenceArguments
from utils.helpers import get_output_file_name, parse_experiment_config


@dataclass
class SubmitArguments:
    """
    Arguments pertaining to submitting an inference experiment.
    """

    ################ 
    ## SLURM
    ################ 

    use_slurm: bool = field(
        default=True,
        metadata={"help": "If set to True, submission command uses sbatch with relevant slurm commands"}
    )

    ntasks: int = field(
        default=1,
        metadata={"help": "SLURM ntasks"}
    )

    cpus_per_task: int = field(
        default=1,
        metadata={"help": "SLURM cpus-per-task"}
    )
    
    gres: str = field(
        default="gpu:T4:1", #"gpu:A100:1",
        metadata={"help": "SLURM gres"}
    )

    mem: str = field(
        default="60GB",
        metadata={"help": "SLURM mem"}
    )

    time: str = field(
        default="04:00:00",
        metadata={"help": "SLURM time"}
    )

    log_file: str = field(
        default="",
        metadata={"help": "SLURM log file path"}
    )

    dry_run: bool = field(
        default=False,
        metadata={"help": "If set to True, submission command is printed to stdout but not executed"}
    )

    debug: bool = field(
        default=False,
        metadata={"help": "If set to True, submission command is executed on dummy script"}
    )

# @dataclass
# class ExperimentArguments:


def slurm_is_available():
    out = subprocess.run(["sinfo"], capture_output=True, shell=True)
    return out.returncode == 0

if __name__ == "__main__":
    
    hf_parser = HfArgumentParser((InferenceArguments, SubmitArguments))

    if sys.argv[1].endswith(".json"):
        # If we pass only a json file as the first argument,
        # we parse it to get our arguments.
        args, s_args = hf_parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        # We also parse the remaining arguments that may be specified, orverriding the ones in the json file.
        remaining_args = sys.argv[2:]
        for i in range(0, len(remaining_args), 2):
            key = remaining_args[i].lstrip('-').replace('-', '_')
            value = remaining_args[i+1]
            if key in args.__dict__:
                args.__dict__[key] = value
            elif key in s_args.__dict__:
                s_args.__dict__[key] = value
            else:
                raise ValueError(f"Unrecognized argument: {key}")

    else:
        args, s_args = hf_parser.parse_args_into_dataclasses()

    if not s_args.use_slurm or not slurm_is_available():
        PREFIX = 'bash ' # will execute the directly
    else: # will submit a slurm job
        PREFIX = f'sbatch ' \
                f'--ntasks={s_args.ntasks} ' \
                f'--cpus-per-task={s_args.cpus_per_task} ' \
                f'--mem={s_args.mem} ' \
                f'--time={s_args.time} '
        
        # infer log file path
        if not s_args.log_file:
            s_args.log_file = get_output_file_name(args, ext=".log")
            # m = Path(args.model_name_or_path).name
            # t = Path(args.input_file).name.replace('.', '-')        
            # log_file = f"logs/{m}_{t}_{args.few_shot_n}_{args.n_refs}.log"
            # s_args.log_file = log_file
            PREFIX += f'--output="{s_args.log_file}" '

        if s_args.gres:
            PREFIX += f'--gres={s_args.gres} '

    SCRIPT = 'slurm_scripts/run_inference_on_t4.sh '
    
    if s_args.debug:
        SCRIPT = 'slurm_scripts/run_dummy.sh '
    
    # if 'llama' in args.model_name_or_path.lower():
    #     SCRIPT = 'slurm_scripts/run_inference_on_a100_llama.sh '
    
    if 'a100' in s_args.gres.lower():
        SCRIPT = 'slurm_scripts/run_inference_on_a100.sh '
    
    SUFFIX = f'--model_name_or_path "{args.model_name_or_path}" ' \
                f'--max_new_tokens {args.max_new_tokens} ' \
                f'--max_memory {args.max_memory} ' \
                f'--batch_size {args.batch_size} ' \
                f'--num_beams {args.num_beams} ' \
                f'--num_return_sequences {args.num_return_sequences} ' \
                f'--seed {args.seed} ' \
                f'--do_sample {args.do_sample} ' \
                f'--top_p {args.top_p} ' \
                f'--temperature {args.temperature} ' \
                f'--examples "{args.examples}" ' \
                f'--input_file "{args.input_file}" ' \
                f'--n_refs {args.n_refs} ' \
                f'--few_shot_n {args.few_shot_n} ' \
                f'--prompt_json "{args.prompt_json}" ' \
                f'--prompt_prefix "{args.prompt_prefix}" ' \
                f'--prompt_suffix "{args.prompt_suffix}" ' \
                f'--prompt_format "{args.prompt_format}" ' \
                f'--prompt_template "{args.prompt_template}" ' \
                f'--source_field "{args.source_field}" ' \
                f'--target_field "{args.target_field}" ' \
                f'--output_dir "{args.output_dir}" '

    full_command = PREFIX + SCRIPT + SUFFIX
    print()
    print(full_command)
    print()
    if not s_args.dry_run:
        # os.system(full_command)
        subprocess.run(full_command, shell=True)
