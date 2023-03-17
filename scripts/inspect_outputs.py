#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __Author__ = 'Tannon Kew'
# __Email__ = 'kew@cl.uzh.ch
# __Date__ = '2023-03-03'

"""

Simple script to randomly inspect model outputs. 
Expects a JSONL input file with items containing input prompts and model outputs.

Example usage:

    python -m scripts.view_outputs data/outputs/bloom-560m/turk-test_turk-tune_2_1_489.jsonl

"""

import argparse
import random
from typing import Dict

from utils import iter_lines, pretty_print_instance

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', type=str, help='Path to the file containing model generated outputs')
    parser.add_argument('--seed', default=None, help='Random seed')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    
    if not args.seed:
        args.seed = random.randint(0, 100000)
    print(f'Using seed: {args.seed}')
    random.seed(int(args.seed))
        
    lines = [line for line in iter_lines(args.infile)]
    random.shuffle(lines)

    for line in lines:
        pretty_print_instance(line)
        cont = input('Press enter to continue, or q to quit: ')
        if cont == 'q':
            break

