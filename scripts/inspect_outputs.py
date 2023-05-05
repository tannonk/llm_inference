#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __Author__ = 'Tannon Kew'
# __Email__ = 'kew@cl.uzh.ch
# __Date__ = '2023-03-03'

"""

Simple script to randomly inspect model outputs. 
Expects a JSONL input file with items containing input prompts and model outputs.

Example usage:

    To inspect samples from a specific model:
    python -m scripts.inspect_outputs --infile resources/outputs/bloom-560m/asset-test_asset-valid_p0_random_fs3_nr1_s287.jsonl

    To inspect samples from different models:
    python -m scripts.inspect_outputs --seed 287 --models bloom,llama-7b,opt-13b --num_examples 2

"""

import argparse
import glob
import random
from tabulate import tabulate
from typing import Dict
from utils.helpers import iter_lines, pretty_print_instance


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', default=None, type=str, help='Path to the file containing model generated outputs')
    parser.add_argument('--seed', default=None, help='Random seed')
    parser.add_argument('--prompt_id', default="p0", help='Id of the input prompt')
    parser.add_argument('--models', default=None, help='List of models to inspect random samples')
    parser.add_argument('--test_set', default="asset", help='Test set name')
    parser.add_argument('--num_examples', type=int, default=1, help='Number of examples for each models list')
    return parser.parse_args()


def peek_outputs(args):
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


def inspect_models(args):
    seed = int(args.seed)
    for _ in range(0, args.num_examples):
        random.seed(seed)
        get_models_data(args)
        seed = seed + 1


def get_models_data(args):
    models = args.models.split(",")
    index = -1
    complex_line = ""
    outputs = []

    for m in models:
        file = glob.glob(f"resources/outputs/{m}/*{args.test_set}*{args.prompt_id}*{args.seed}*jsonl")
        lines = [line for line in iter_lines(file[0])]
        if index == -1:
            index = random.randint(0, len(lines))
        item = lines[index]
        complex_line = item["source"]
        outputs.append([f"{m}:", item["model_output"]])

    show_results(complex_line, outputs)


def show_results(complex_line, outputs):
    results = [["complex:", complex_line]] + outputs
    pretty_result = tabulate(results, headers=['Model', 'Sentence(s)'])
    print(f"{pretty_result}\n")


if __name__ == '__main__':
    args = get_args()

    if args.infile:
        peek_outputs(args)
    elif args.models:
        inspect_models(args)
    else:
        print("No valid arguments were defined, the script needs either a file (--infile) "
              "or a list of models (--models) to inspect.")
