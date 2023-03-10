#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __Author__ = 'Tannon Kew'
# __Email__ = 'kew@cl.uzh.ch
# __Date__ = '2023-03-03'


"""

Generic functions for file handling

"""

import re
import json
import logging
import hashlib
import pprint
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Generator

from llm_inference import InferenceArguments

logger = logging.getLogger(__name__)

def iter_text_lines(file: Union[str, Path]) -> Generator[str, None, None]:
    """Generator that yields lines from a regular text file."""
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            else:
                yield line

def iter_json_lines(file: Union[str, Path]) -> Generator[Dict, None, None]:
    """Fetch dictionary-object lines from a JSONL file"""
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            else:
                yield json.loads(line)

def iter_lines(file: Union[str, Path]) -> Generator[Union[str, Dict], None, None]:
    """Wraps `iter_text_lines` and `iter_json_lines` to fetch lines from file"""
    if str(file).endswith(".jsonl") or str(file).endswith(".json"):
        return iter_json_lines(file)
    else:
        return iter_text_lines(file)

def load_few_shot_prompts(fsp_file: Union[str, Path]) -> List[str]:
    """Returns a list of few-shot prompts."""
    fsprompts = [l for l in iter_lines(fsp_file)]
    logger.info(f"Loaded {len(fsprompts)} few-shot prompts")
    return fsprompts

def load_prompts(p_file: Union[str, Path]) -> List[str]:
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
        if len(fsprompts) != len(prompts):
            raise RuntimeError("Number of few-shot prompts must be 1 or equal to number of prompts!")
        return [fsprompts[i] + prompts[i] for i in range(len(prompts))]

def iter_json_batches(file: str, batch_size: int = 3) -> Generator[List[Dict], None, None]:
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

def iter_text_batches(file: Union[str, Path], batch_size: int = 3) -> Generator[List[str], None, None]:
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

def iter_batches(file: Union[str, Path], batch_size: int = 3) -> Generator[Union[List[str], List[Dict]], None, None]:
    """Wraps `iter_text_batches` and `iter_json_batches` to fetch batched lines from file"""
    if str(file).endswith(".jsonl") or str(file).endswith(".json"):
        return iter_json_batches(file, batch_size)
    else:
        return iter_text_batches(file, batch_size)


def get_output_file_name(args: InferenceArguments, ext: str = ".jsonl") -> str:
    """Given all inference arguments, generate output filename for consistency"""
    
    model_name = re.sub("[\.\_]", "-", Path(args.model_name_or_path).name) # 'bigscience/bloom-1b1 -> bloom-1b1
    
    test_set = re.sub("[\.\_]", "-", Path(args.input_file).stem) # data/asset/dataset/asset.test.orig -> asset-test
    
    examples = re.sub("[\.\_]", "-", Path(args.examples).stem) # data/asset/dataset/asset.valid -> asset-valid
    
    if args.prompt_json is not None:
        prompt_id = re.sub("[\.\_]", "-", Path(args.prompt_json).stem) # prompts/ex1.json-> ex1
    else: # generate an on-the-fly 'prompt id' based on the prompt prefix and format used
        prompt_hash = f"{hashlib.sha1(args.prompt_prefix.encode('UTF-8')).hexdigest()[:8]}"
        prompt_id = prompt_hash + '-' + re.sub("[\.\_]", "-", args.prompt_format)

    if examples == test_set: # this may be necessary to avoid if no validation set is available
        raise RuntimeError("Few-shot prompt examples should not be the same as the test instances!")

    output_file = Path(f"{args.output_dir}") / f"{model_name}" / f"{test_set}_{examples}_" \
                                                                f"{prompt_id}_" \
                                                                f"{args.few_shot_n}_" \
                                                                f"{args.n_refs}_" \
                                                                f"{args.seed}{ext}"

    logger.info(f"Model outputs will be written to {output_file}")
    if Path(output_file).exists():
        logger.warning(f"Output file {output_file} already exists! Overwriting...")
    else: # create directory path if necessary
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    return str(output_file)

def persist_args(args: InferenceArguments) -> None:
    """Writes inference args to file for reference / experiment tracking.
    The file name is inferred from the name of the output_file."""
    
    if args.output_file != "stdout":
        inference_args_file = Path(args.output_file).parent / f'{Path(args.output_file).stem}_args.json'
    
        with open(str(inference_args_file), "w", encoding ="utf8") as outf:
            json.dump(args.__dict__, outf, ensure_ascii=False, indent=4)

        logger.info(f"Inference args written to {inference_args_file}")
    
    logger.info(f"Inference args = {args}")

    return

def serialize_to_jsonl(inputs: List[str], outputs: List[List[str]]) -> Generator[str, None, None]:
    """Generator function to write each model output as a json object line by line"""
    for input_sequence, output_sequences in zip(inputs, outputs):
        yield json.dumps({"input_prompt": input_sequence, "model_output": r'\t'.join(output_sequences).strip()}, ensure_ascii=False)

def parse_experiment_config(config_file: str) -> Dict:
    with open(config_file, 'r') as f:
        data = json.load(f)
    return data

def pretty_print_instance(example: Dict) -> None:
    
    input_str = re.sub(r'\\n\\n', '\n\n', example.get('input_prompt'))
    input_str = re.sub(r'\\n', '\n', input_str)

    print(f"Input:\n")
    # pprint.pprint(input_str)
    print(input_str)
    print(f"\nOutput:\n")
    # pprint.pprint(example.get('model_output'))
    print(example.get('model_output'))
    print("="*80)


if __name__ == "__main__":
    pass