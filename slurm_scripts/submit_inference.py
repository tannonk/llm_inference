#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
    --examples "data/asset/dataset/valid.jsonl" \
    --input_file "data/asset/dataset/asset.test.orig" \
    --prompt_prefix "I want you to replace my complex sentence with simple sentence(s). Keep the meaning same, but make them simpler."

"""


import os, sys
import argparse
from dataclasses import dataclass, field

from transformers import HfArgumentParser
from llm_inference import InferenceArguments
from utils import get_output_file_name


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
        metadata={"help": ""}
    )

    cpus_per_task: int = field(
        default=1,
        metadata={"help": ""}
    )
    
    gres: str = field(
        default=None, #"gpu:A100:1",
        metadata={"help": ""}
    )

    mem: str = field(
        default="300GB",
        metadata={"help": ""}
    )

    time: str = field(
        default="04:00:00",
        metadata={"help": ""}
    )

    log_file: str = field(
        default=None,
        metadata={"help": ""}
    )

    experiment_config: str = field(
        default=None,
        metadata={"help": ""}
    )

def parse_experiment_config(config_file):
    with open(config_file, 'r') as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    
    hf_parser = HfArgumentParser((InferenceArguments, SubmitArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args, s_args = hf_parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        args, s_args = hf_parser.parse_args_into_dataclasses()


    if not s_args.use_slurm:
        PREFIX = 'bash '
        SCRIPT = 'slurm_scripts/run_dummy.sh'

    else:
        PREFIX = f'sbatch ' \
                f'--ntasks={s_args.ntasks} ' \
                f'--cpus-per-task={s_args.cpus_per_task} ' \
                f'--mem={s_args.mem} ' \
                f'--time={s_args.time} '
        
        # infer log file path
        if s_args.log_file is None:
            s_args.log_file = get_output_file_name(args, ext=".log")
            # m = Path(args.model_name_or_path).name
            # t = Path(args.input_file).name.replace('.', '-')        
            # log_file = f"logs/{m}_{t}_{args.few_shot_n}_{args.n_refs}.log"
            # s_args.log_file = log_file
            PREFIX += f'--output="{s_args.log_file}" '

        if s_args.gres:
            PREFIX += f'--gres={s_args.gres} '

            if 't4' in s_args.gres.lower():
                SCRIPT = 'slurm_scripts/run_inference_on_t4.sh '
            elif 'a100' in s_args.gres.lower():
                SCRIPT = 'slurm_scripts/run_inference_on_a100.sh '
        else: # debug
            SCRIPT = 'slurm_scripts/run_dummy.sh '

    # infer output file for generations --> moved to inference.py
    # if args.output_file is None:
    #     m = Path(args.model_name_or_path).name
    #     t = Path(args.input_file).name.replace('.', '_')        
    #     output_file = f"data/outputs/{m}/{t}_{args.few_shot_n}_{args.n_refs}_{args.seed}.jsonl"
    #     args.output_file = output_file
    
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
                f'--prompt_prefix "{args.prompt_prefix}" ' \
                f'--output_dir "{args.output_dir}" '

    full_command = PREFIX + SCRIPT + SUFFIX
    print()
    print(full_command)
    print()
    os.system(full_command)