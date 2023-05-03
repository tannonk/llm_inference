#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Tannon Kew
# Date: 2023-04-13

"""
Helper script for checking logs and outputs of inference runs.

Example call:

    python scripts/check_logs.py

NOTE: this script assumes the location of the outputs directory is: resources/outputs/
"""

import sys
import logging
from tqdm import tqdm
import argparse
import json
import subprocess
from typing import List
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

eval_header = "bleu;sari;fkgl;pbert_ref;rbert_ref;fbert_ref;pbert_src;rbert_src;fbert_src;ppl_mean;ppl_std;lens;lens_std;intra_dist1;intra_dist2;inter_dist1;inter_dist2;Compression ratio;Sentence splits;Levenshtein similarity;Exact copies;Additions proportion;Deletions proportion;Lexical complexity score;file_id"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_configs", type=str, default="exp_configs/cluster/", dest="EXP_TO_RUN",
                        help="Path to model configuration templates for running experiments")
    parser.add_argument("--outputs_dir", type=str, default="resources/outputs/", dest="EXP_READY",
                        help="Path to results from the configuration templates")
    parser.add_argument("--do_run", action="store_true", help="Run the evaluation command if evaluation results are missing")
    parser.add_argument("--verbose", action="store_true", help="")
    return parser.parse_args()

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def line_count(filepath: Path, verbose: bool = False) -> int:
    if not filepath.exists():
        print(f'{bcolors.FAIL}File not found {filepath} {bcolors.ENDC}')
    else:
        if verbose:
            with open(filepath, 'r', encoding='utf8') as f:
                lc = sum(1 for _ in f)
            logger.info(f'{lc} lines found in {filepath}')
    return 

def check_eval_file(filepath: Path, verbose: bool = False) -> None:
    if not filepath.exists():
        print(f'{bcolors.FAIL}File not found {filepath} {bcolors.ENDC}')
    else:
        with open(filepath, 'r', encoding='utf8') as f:
            if f.readlines()[0].strip() != eval_header:
                print(f'{bcolors.FAIL}Header mismatch in {filepath} ({datetime.fromtimestamp(filepath.stat().st_mtime)}) {bcolors.ENDC}')
    return

def check_log_file(filepath: Path, do_run: bool = False, verbose: bool = False) -> None:
    with open(filepath, 'r', encoding='utf8') as f:
        log = f.read()
        # check for errors explitly
        if 'Traceback' in log:
            print(f'{bcolors.FAIL}Traceback found in {filepath} ({datetime.fromtimestamp(filepath.stat().st_mtime)}) {bcolors.ENDC}')
            print('*********')
            # print sample of log with context following error message
            print('\t', log[log.find('Traceback'):100])
            print('*********')
        elif 'slurmstepd: error:' in log:
            print(f'{bcolors.FAIL}SLURM error found in {filepath} ({datetime.fromtimestamp(filepath.stat().st_mtime)}) {bcolors.ENDC}')
            print('*********')
            # print sample of log with context following error message
            print('\t', log[log.find('slurmstepd: error:'):])
            print('*********')
        elif not 'Finished inference' in log:
            print(f'{bcolors.FAIL}Failed to complete inference in {filepath} ({datetime.fromtimestamp(filepath.stat().st_mtime)}) {bcolors.ENDC}')
            print('\t', log[-200:])
        elif not 'Wrote results to' in log:
            print(f'{bcolors.FAIL}Failed to write results in {filepath} ({datetime.fromtimestamp(filepath.stat().st_mtime)}) {bcolors.ENDC}')
            print('\t', log[-200:])
            # if there's no evidence of evaluation results in the log, get the command to run evaluation
            eval_command = get_eval_command(filepath, verbose=verbose, do_run=do_run)
            if not do_run:
                print('*** You could run the following command: ***')
                print(eval_command)
                print()
            if do_run:
                print('*** Attempting to run the following command: ***')
                print(eval_command)
                run_command(eval_command, verbose=verbose)
                print()
        else: # seems to have completed successfully
            if verbose:
                logger.info(f'{bcolors.OKGREEN}No errors found in {filepath} ({datetime.fromtimestamp(filepath.stat().st_mtime)}) {bcolors.ENDC}')
    return

def collect_config_files(config_dir: Path, verbose: bool = False) -> List[Path]:
    config_files = []
    for filepath in sorted(list(config_dir.glob('*.json'))):
        config_files.append(filepath.stem)
    return config_files

def get_eval_command(filepath: Path, verbose: bool = False) -> None:

    with open(filepath.with_suffix('.json'), 'r', encoding='utf8') as f:
        args = json.load(f)
    model_config = f'exp_configs/cluster/{Path(args["output_file"]).parent.name}.json' # 'exp_configs/cluster/opt-13b.json'

    eval_command = f'python -m run {model_config} ' \
        f'--seed {args["seed"]} ' \
        f'--input_file {args["input_file"]} ' \
        f'--output_dir {args["output_dir"]} ' \
        f'--examples {args["examples"]} ' \
        f'--prompt_json {args["prompt_json"]} ' \
        f'--example_selector {args["example_selector"]} ' \
        f'--n_refs {args["n_refs"]} ' \
        f'--few_shot_n {args["few_shot_n"]} ' \
        f'--do_inference False --do_evaluation True --dry_run False '

    return eval_command

def run_command(command):
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    if result.returncode == 0:
        try:
            job_id = int(result.stdout.strip().split()[-1])
        except:
            job_id = None
        print(f"Evaluation job id: {job_id}")
    else: 
        print(result.stderr)
        raise ValueError(f"Evaluation job submission failed.")
    return 

def check_files(log_file: Path, outputs_file: Path, eval_file: Path, args_file: Path, do_run: bool = False, verbose: bool = False) -> None:
    """
    runs all checks on a given set of files

    NOTE: currently no check for the args file
    TODO: handle missing log files
    """
    check_log_file(log_file, do_run=do_run, verbose=args.verbose)
    check_eval_file(eval_file, verbose=args.verbose)
    line_count(outputs_file, verbose=args.verbose)
    return
    

if __name__ == '__main__':
    
    args = parse_args()
    # collect config files in order to check for missing or not yet run configs
    configs = collect_config_files(Path(args.EXP_TO_RUN))
    # print(configs)

    data_dir = Path(args.EXP_READY)

    c = 0
    for model_dir in data_dir.iterdir():
        if model_dir.name.lower() in configs:
            # logger.info(f'{bcolors.OKBLUE}Checking {model_dir} {bcolors.ENDC}')
            # remove the config from the list if the output dir exists
            configs.remove(model_dir.name.lower())

            # check for problematic log files and output files
            for filepath in tqdm(model_dir.glob('*.jsonl')):
                log_file = filepath.with_suffix('.log')
                outputs_file = filepath.with_suffix('.jsonl')
                eval_file = filepath.with_suffix('.eval')
                args_file = filepath.with_suffix('.json')
                check_files(log_file, outputs_file, eval_file, args_file, 
                            do_run=args.do_run, verbose=args.verbose)
        
    # if there are any configs left, they have not been run yet
    if len(configs) > 0:
        print(f'{bcolors.WARNING}No output dirs found for the following models:{bcolors.ENDC} {configs}')

    print(f'{bcolors.OKGREEN}Checked {c} log files {bcolors.ENDC}')