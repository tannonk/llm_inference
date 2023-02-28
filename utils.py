#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Generic functions for file handling

"""

import json
import logging
import logging
from typing import List, Dict, Tuple, Optional, Union

logger = logging.getLogger(__name__)

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

def iter_json_batches(file: str, batch_size: int = 3):
    """Fetch batched lines from jsonl file"""
    current_batch = []
    c = 0
    for line in iter_json_lines(file):
        current_batch.append(line)
        c += 1
        if c == batch_size and len(current_batch) > 0:
            yield current_batch
            # reset vars for next batch
            c = 0
            current_batch = []    
    if len(current_batch) > 0:
        yield current_batch # don't forget the last one!

def iter_batches(file: str, batch_size: int = 3):
    """Fetch batched lines from file"""
    current_batch = []
    c = 0
    for line in iter_lines(file):
        current_batch.append(line)
        c += 1
        if c == batch_size and len(current_batch) > 0:
            yield current_batch
            # reset vars for next batch
            c = 0
            current_batch = []    
    if len(current_batch) > 0:
        yield current_batch # don't forget the last one!

def serialize_to_jsonl(inputs: List[str], outputs: List[str]):
    for input_sequence, output_sequences in zip(inputs, outputs):
        yield {"input_prompt": input_sequence, "model_output": '\t'.join(output_sequences).strip()}

if __name__ == "__main__":
    pass