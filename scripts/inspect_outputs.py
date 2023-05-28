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
    python -m scripts.inspect_outputs --seed 489 --models_file models.csv --num_examples 5

    Example: models.csv
    openai-text-davinci-003,p1
    t5-small-lm-adapt,p0
    llama-7b,p1

"""

from pathlib import Path
import pandas as pd
import argparse
import glob
import random
from tabulate import tabulate
from typing import Dict
from utils.helpers import iter_lines, pretty_print_instance


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', default=None, type=str, help='Path to the file containing model generated outputs')
    parser.add_argument('--seed', default=287, type=int, help='Random seed')
    parser.add_argument('--prompt_id', default="p0", help='Id of the input prompt')
    parser.add_argument('--models', default=None, help='List of models to inspect random samples')
    parser.add_argument('--test_set', default="asset", help='Test set name')
    parser.add_argument('--strategy', default="random", help='Strategy used to select few-shot examples')
    parser.add_argument('--fs', default=3, type=int, help='Number of few-shot examples')
    parser.add_argument('--num_examples', type=int, default=1, help='Number of examples for each models list')
    parser.add_argument('--models_file', default=None, help='List of models and prompts to inspect random samples')
    parser.add_argument('--output_dir', default="resources/outputs", help='Path to the directory containing model outputs')
    return parser.parse_args()


def peek_outputs(args):
    if not args.seed:
        args.seed = random.randint(0, 100000)
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
    print(f'Using seed: {args.seed}')
    print(f'\n=== Settings ==='
          f'\n- seed: {args.seed}'
          f'\n- test_set: {args.test_set}'
          f'\n- strategy: {args.strategy}'
          f'\n- fs: {args.fs}'
          f'\n================\n')
    models = get_models(args)
    for _ in range(0, args.num_examples):
        random.seed(seed)
        get_models_data(args, models)
        seed = seed + 1

def reconstruct_file_path(args, model, prompt_id):
    file_path = Path(args.output_dir) / f"{model}"
    file_name = (f"{args.test_set}*_{prompt_id}_{args.strategy}_fs{args.fs}_nr1_s{args.seed}.jsonl")
    file = glob.glob(str(file_path / file_name))
    if len(file) == 0:
            raise Exception(f"There are no model outputs for {str(file_path / file_name)}")
    elif len(file) > 1:
        Warning(f"There is more than one file for the specified args! Using {file[0]}")
        file = [file[0]]
    return file

def get_models_data(args, models):
    index = -1
    complex_line = ""
    outputs = []

    for model, prompt_id in models:
        file = reconstruct_file_path(args, model, prompt_id)
        lines = [line for line in iter_lines(file[0])]
        if index == -1:
            index = random.randint(0, len(lines))
        item = lines[index]
        complex_line = item["source"]
        references = item["references"]
        outputs.append([f"{model}-{prompt_id}:", item["model_output"]])

    show_results(complex_line, references, outputs)


def get_models(args):
    models = []
    if args.models_file:
        with open(args.models_file) as f1:
            for line in f1:
                line = line.strip()
                model, prompt = line.split(",")
                models.append([model, prompt])
    elif args.models:
        models = args.models.split(",")
        models = [[m, args.prompt_id] for m in models]

    return models


def show_results(complex_line, references, outputs):
    results = [["complex:", complex_line], ["references:", references[0]]] + outputs
    # pretty_result = tabulate(results, headers=['Model', 'Sentence(s)'])
    # print(f"{pretty_result}\n")
    # hack using pandas to print the table in a more compact way
    print(pd.DataFrame(results).T.set_index(0, drop=True).stack().unstack(0).reset_index(drop=True).to_csv(sep='\t', index=False))


if __name__ == '__main__':
    args = get_args()

    if args.infile:
        peek_outputs(args)
    elif args.models or args.models_file:
        inspect_models(args)
    else:
        print("No valid arguments were defined, the script needs either a file (--infile) "
              "or a list of models (--models) to inspect.")
