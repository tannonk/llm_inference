#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from pathlib import Path
import argparse
import logging
from typing import List, Dict, Tuple, Optional, Union
import json
from llm_inference import InferenceArguments, LLM
from transformers import (
    HfArgumentParser
)
# logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
logger = logging.getLogger(__name__)
# logger.info(args)

# def parse_custom_args():
#     parser = argparse.ArgumentParser()
#     # parser.add_argument('-i', '--input_file', required=True, type=str, help='filepath to input file')
#     # parser.add_argument('--fsp_file', required=False, type=str, help='filepath to few-shot prompts')
#     # parser.add_argument('-o', '--output_file', required=False, type=str, help='filepath to output file')
#     # parser.add_argument('-v', '--verbose', action='store_true')
#     # parser.add_argument('-d', '--debug', action='store_true')
#     # parser.add_argument('-c', '--config', required=False, type=str, help='Model name with predefined config to use for inference')
#     return parser.parse_known_args()

def iter_lines(file):
    """Generator that yields lines from a regular text file."""
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            else:
                yield line

def iter_json_lines(file: str):
    """Fetch dictionary-object lines from a JSONL file"""
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            else:
                yield json.loads(line)

def load_few_shot_prompts(fsp_file: str) -> List[str]:
    """Returns a list of few-shot prompts."""
    fsprompts = [l for l in iter_lines(fsp_file)]
    logger.info(f"Loaded {len(fsprompts)} few-shot prompts")
    return fsprompts

def load_prompts(p_file: str) -> List[str]:
    """Returns a list of prompts."""
    prompts = [l for l in iter_lines(p_file)]
    logger.info(f"Loaded {len(prompts)} prompts")
    return prompts

def merge_prompts(prompts: List[str], fsprompts: Optional[List[str]] = None) -> List[str]:
    if not fsprompts: # zero-shot setting
        return prompts
    elif len(fsprompts) == 1: # every input is concatenated with the same fsprompt
        return [fsprompts[0] + prompt for prompt in prompts]
    else:
        assert len(fsprompts) == len(prompts)
        return [fsprompts[i] + prompts[i] for i in range(len(prompts))]

def iter_batches(file: str, batch_size: int = 3):
    """Fetch batched lines from file"""
    current_batch = []
    c = 0
    for line in iter_json_lines(file):
        current_batch.append(line)
        c += 1
        if c == batch_size:
            yield current_batch
            # reset vars for next batch
            c = 0
            current_batch = []    
    yield current_batch # don't forget the last one!

def prepare_inputs(examples: List[Dict], few_shot_n: int = 0, delimiter: str = '\n\n', seed: int = 42) -> List[str]:
    """
    prepares few-shot (or zero-shot) inputs for LLM inference
    """
    # set random seed for reproducibility
    random.seed(seed)

    inputs = []
    for ex in examples:
        if few_shot_n == 0:
            inputs.append(ex['src'])
        elif isinstance(ex['examples'], list):
            assert few_shot_n < len(ex['examples']), f"few_shot_n ({few_shot_n}) can not be greater than the number of available examples ({len(ex['examples'])})"
            input_str = delimiter.join(random.sample(ex['examples'], few_shot_n)) + delimiter + ex['src']
            inputs.append(input_str)
        else:
            raise NotImplementedError(f'Expected examples to be a list of examples, but got {type(ex["examples"])}')
                # assert delimiter in ex['examples'], f"Delimiter {delimiter} is not consistent between examples and src"
                # inputs.append(f"{ex['examples']}{delimiter}{ex['src']}")
    return inputs


if __name__ == '__main__':

    parser = HfArgumentParser((InferenceArguments))
    args = parser.parse_args_into_dataclasses()[0]
    
    # prompts = load_few_shot_prompts(args.input_file)
    # fsprompts = load_few_shot_prompts(args.fsp_file) if args.fsp_file else None
    # prompts = merge_prompts(prompts, fsprompts)

    # print(prompts)
    # facebook/opt-30b

    llm = LLM(args.model_name_or_path, args.max_memory)

    for batch_prompts in iter_batches(args.input_file, args.batch_size):
        print(len(batch_prompts))
        # print(batch_prompts)
        inputs = prepare_inputs(batch_prompts, args.few_shot_n, args.delimiter, args.seed)
        outputs = llm.generate_from_model(inputs, args)
    
        outputs = llm.postprocess_model_outputs(inputs, outputs, args.delimiter)
        print(outputs)