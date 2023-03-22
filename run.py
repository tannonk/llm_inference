#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __Author__ = 'Tannon Kew'
# __Email__ = 'kew@cl.uzh.ch
# __Date__ = '2023-03-09'

"""

This is a wrapper to facilitate the execution of experiments 
either on a slurm cluster or directly on a local machine.

Example Call:

python -m run \
    --use_slurm True \
    --ntasks 1 \
    --cpus_per_task 1 \
    --gres gpu:T4:1 \
    --mem 32GB \
    --time 00:30:00 \
    --batch_size 8 \
    --seed 489 \
    --model_name_or_path bigscience/bloom-560m \
    --examples data/asset/dataset/asset.valid.jsonl \
    --input_file data/asset/dataset/asset.test.jsonl \
    --prompt_json prompts/p0.json

Alternatively, you can pass a json file in position 1 with some or all of the arguments:

python -m run exp_configs/bloom-560m.json \
    --seed 489 \
    --examples data/asset/dataset/asset.valid.jsonl \
    --input_file data/asset/dataset/asset.test.jsonl \
    --prompt_json prompts/p0.json \
    --n_refs 1 --few_shot_n 3

This script will produce the following files:

    - <output_file>.jsonl: The predictions of the model on the input file.
    - <output_file>.json: Command line arguments used for the inference run.
    - <output_file>.log: Log file of the inference run.
    - <output_file>.eval: Log file of the automatic evaluation with results. 

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
    

    mem: str = field(
        default="12GB",
        metadata={"help": "SLURM mem"}
    )

    time: str = field(
        default="00:30:00",
        metadata={"help": "SLURM time"}
    )

    n_gpus: int = field(
        default=1,
        metadata={"help": "Number of GPUs to use"}
    )

    gpu_type: str = field(
        default="T4",
        metadata={"help": "GPU type"}
    )

    log_file: str = field(
        default="",
        metadata={"help": "SLURM log file path"}
    )

    debug: bool = field(
        default=False,
        metadata={"help": "If set to True, submission command is executed on dummy script"}
    )

    do_inference: bool = field(
        default=True,
        metadata={"help": "If set to True, inference is executed"}
    )

    do_evaluation: bool = field(
        default=True,
        metadata={"help": "If set to True, evaluation is executed"}
    )


def slurm_is_available():
    out = subprocess.run(["sinfo"], capture_output=True, shell=True)
    return out.returncode == 0

# https://discuss.pytorch.org/t/it-there-anyway-to-let-program-select-free-gpu-automatically/17560/13
def run_cmd(cmd):
    out = (subprocess.check_output(cmd, text=True, shell=True))[:-1]
    return out

def get_free_gpu_indices():
    out = run_cmd('nvidia-smi -q -d Memory | grep -A4 GPU')
    out = (out.split('\n'))[1:]
    out = [l for l in out if '--' not in l]

    total_gpu_num = int(len(out)/5)
    gpu_bus_ids = []
    for i in range(total_gpu_num):
        gpu_bus_ids.append([l.strip().split()[1] for l in out[i*5:i*5+1]][0])

    out = run_cmd('nvidia-smi --query-compute-apps=gpu_bus_id --format=csv')
    gpu_bus_ids_in_use = (out.split('\n'))[1:]
    gpu_ids_in_use = []

    for bus_id in gpu_bus_ids_in_use:
        gpu_ids_in_use.append(gpu_bus_ids.index(bus_id))

    return [i for i in range(total_gpu_num) if i not in gpu_ids_in_use]


if __name__ == "__main__":
    
    hf_parser = HfArgumentParser((InferenceArguments, SubmitArguments))

    if sys.argv[1].endswith(".json"):
        # If we pass only a json file as the first argument,
        # we parse it to get our arguments.
        i_args, s_args = hf_parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        # We also parse the remaining arguments that may be specified, orverriding the ones in the json file.
        remaining_args = sys.argv[2:]
        for i in range(0, len(remaining_args), 2):
            key = remaining_args[i].lstrip('-').replace('-', '_')
            value = remaining_args[i+1]
            if key in i_args.__dict__:
                i_args.__dict__[key] = value
            elif key in s_args.__dict__:
                s_args.__dict__[key] = value
            else:
                raise ValueError(f"Unrecognized argument: {key}")

    else:
        i_args, s_args = hf_parser.parse_args_into_dataclasses()

    output_file = get_output_file_name(i_args, ext=".jsonl")
    log_file = get_output_file_name(i_args, ext=".log") if not s_args.log_file else s_args.log_file

    # Inference
    if s_args.do_inference:
        # Check if slurm is available. If not, will execute directly
        if not s_args.use_slurm or not slurm_is_available():

            prefix = 'nohup bash '

            # Set GPU resources
            avail_gpus = get_free_gpu_indices()
            if len(avail_gpus) < s_args.n_gpus:
                raise ValueError(f"Only {len(avail_gpus)} GPUs available, but {s_args.n_gpus} requested.")
            else:
                avail_gpus = ','.join([str(i) for i in avail_gpus[:s_args.n_gpus]])                   
                prefix = f'CUDA_VISIBLE_DEVICES={avail_gpus} ' + prefix

            suffix = f' >| {log_file} &'
        
        else:

            prefix = f'sbatch ' \
                    f'--ntasks={s_args.ntasks} ' \
                    f'--cpus-per-task={s_args.cpus_per_task} ' \
                    f'--mem={s_args.mem} ' \
                    f'--time={s_args.time} ' \
                    f'--output={log_file} '
                
            # Set GPU resources
            if s_args.n_gpus > 0:
                if s_args.gpu_type:
                    gres = f'gpu:{s_args.gpu_type}:{s_args.n_gpus}'
                else:
                    gres = f'gpu:{s_args.n_gpus}'            
                prefix += f'--gres={gres} '
            
            suffix = ''

        # Set default script on small GPU
        if s_args.debug:
            script = 'slurm_scripts/run_dummy.sh '
        elif s_args.gpu_type and s_args.gpu_type == 'a100':
            script = 'slurm_scripts/run_inference_on_a100.sh '
        else:
            script = 'slurm_scripts/run_inference_on_t4.sh '

    
        args = f'--model_name_or_path "{i_args.model_name_or_path}" ' \
                    f'--max_new_tokens {i_args.max_new_tokens} ' \
                    f'--max_memory {i_args.max_memory} ' \
                    f'--batch_size {i_args.batch_size} ' \
                    f'--num_beams {i_args.num_beams} ' \
                    f'--num_return_sequences {i_args.num_return_sequences} ' \
                    f'--seed {i_args.seed} ' \
                    f'--do_sample {i_args.do_sample} ' \
                    f'--top_p {i_args.top_p} ' \
                    f'--temperature {i_args.temperature} ' \
                    f'--examples "{i_args.examples}" ' \
                    f'--input_file "{i_args.input_file}" ' \
                    f'--n_refs {i_args.n_refs} ' \
                    f'--few_shot_n {i_args.few_shot_n} ' \
                    f'--prompt_json "{i_args.prompt_json}" ' \
                    f'--prompt_prefix "{i_args.prompt_prefix}" ' \
                    f'--prompt_suffix "{i_args.prompt_suffix}" ' \
                    f'--prompt_format "{i_args.prompt_format}" ' \
                    f'--prompt_template "{i_args.prompt_template}" ' \
                    f'--source_field "{i_args.source_field}" ' \
                    f'--target_field "{i_args.target_field}" ' \
                    f'--output_dir "{i_args.output_dir}" ' \
                    f'--output_file "{output_file}"'

        inference_command = prefix + script + args + suffix
        
        print()
        print(inference_command)
        print()
        
        result1 = subprocess.run(inference_command, capture_output=True, text=True, shell=True)
        # Check the status of job A and get the job ID from slurm
        if result1.returncode == 0:
            try:
                job_id1 = result1.stdout.strip().split()[-1]
                print(f"Inference job id: {job_id1}")
            except IndexError:
                pass
        else:
            print(result1.stderr)
            raise ValueError(f"Inference job submission failed")    

    # Evaluate     
    if s_args.do_evaluation: 
        
        # infer log file path
        log_file = get_output_file_name(i_args, ext=".eval") if not s_args.log_file else s_args.log_file

        # check if slurm is available. If not, will execute directly
        if not s_args.use_slurm or not slurm_is_available():

            prefix = 'nohup bash ' # will execute the directly

            # Set GPU resources
            avail_gpus = get_free_gpu_indices()
            if len(avail_gpus) < 1:
                print("[!] No free GPUs. Will attempt to run on GPU 0...")
                avail_gpus = ['0']
            else:
                prefix = f'CUDA_VISIBLE_DEVICES={avail_gpus[0]} ' + prefix

            suffix = f' >| {log_file} &'

        else: # will submit a slurm job

            prefix = f'sbatch '        
            # specify the dependency on inference job if executed
            if s_args.do_inference:
                prefix += f'--dependency=afterok:{job_id1} '

            prefix += f'--output="{log_file}" '

            suffix = ''

        if s_args.debug:
            script = 'slurm_scripts/run_dummy.sh '
        else:
            script = 'slurm_scripts/run_evaluation.sh '

        evaluate_command = prefix + script + f'"{output_file}"' + suffix
        
        print()
        print(evaluate_command)
        print()
        
        result2 = subprocess.run(evaluate_command, capture_output=True, text=True, shell=True)
        
        if result2.returncode == 0:
            try:
                job_id2 = result2.stdout.strip().split()[-1]
                print(f"Evaluation job id: {job_id2}")
            except IndexError:
                pass
        
        else: 
            # Abort inference job
            try:
                subprocess.run(['scancel', job_id1])
                print(result2.stderr)
                raise ValueError(f"Evaluation job submission failed. Aborted {job_id1}.")
            except NameError:
                raise ValueError(f"Evaluation job submission failed.")
        