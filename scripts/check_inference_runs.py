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

def line_count(filepath: Path) -> int:
    if not filepath.exists():
        logger.warning(f'{bcolors.FAIL}File not found {filepath} {bcolors.ENDC}')
    else:
        with open(filepath, 'r', encoding='utf8') as f:
            lc = sum(1 for _ in f)
        logger.info(f'{lc} lines found in {filepath}')
    return 

def check_log_file(filepath: Path) -> None:
    with open(filepath, 'r', encoding='utf8') as f:
        log = f.read()
        if not 'Finished inference' in log:
            logger.warning(f'{bcolors.FAIL}Failed to complete inference in {filepath} ({datetime.fromtimestamp(filepath.stat().st_mtime)}) {bcolors.ENDC}')
            print('\t', log[-200:])
            return
        if not 'Wrote results to' in log:
            logger.warning(f'{bcolors.FAIL}Failed to write results in {filepath} ({datetime.fromtimestamp(filepath.stat().st_mtime)}) {bcolors.ENDC}')
            print('\t', log[-200:])
            return

        # check for errors explitly
        if 'Traceback' in log:
            logger.warning(f'{bcolors.FAIL}Traceback found in {filepath} ({datetime.fromtimestamp(filepath.stat().st_mtime)}) {bcolors.ENDC}')
            print('*********')
            # print sample of log with context following error message
            print('\t', log[log.find('Traceback'):100])
            print('*********')
        elif 'slurmstepd: error:' in log:
            logger.warning(f'{bcolors.FAIL}SLURM error found in {filepath} ({datetime.fromtimestamp(filepath.stat().st_mtime)}) {bcolors.ENDC}')
            print('*********')
            # print sample of log with context following error message
            print('\t', log[log.find('slurmstepd: error:'):])
            print('*********')
        else:
            logger.info(f'{bcolors.OKGREEN}No errors found in {filepath} ({datetime.fromtimestamp(filepath.stat().st_mtime)}) {bcolors.ENDC}')
    return

def collect_config_files(config_dir: Path) -> List[Path]:
    config_files = []
    for filepath in sorted(list(config_dir.glob('*.json'))):
        config_files.append(filepath.stem)
    return config_files

if __name__ == '__main__':
    
    # collect config files in order to check for missing or not yet run configs
    config_dir = Path('exp_configs/cluster/')
    configs = collect_config_files(config_dir)
    # print(configs)

    data_dir = Path('resources/outputs/')

    for model_dir in data_dir.iterdir():
        if model_dir.name.lower() in configs:
            # logger.info(f'{bcolors.OKBLUE}Checking {model_dir} {bcolors.ENDC}')
            # remove the config from the list if the output dir exists
            configs.remove(model_dir.name.lower())

        # check for problematic log files and output files
        for filepath in model_dir.glob('*.log'):
            check_log_file(filepath)            
            outputs_file = filepath.with_suffix('.jsonl')
            line_count(outputs_file)
    
    # if there are any configs left, they have not been run yet
    if len(configs) > 0:
        for config in configs:
            logger.warning(f'{bcolors.WARNING}No output dir found for {config} {bcolors.ENDC}')
