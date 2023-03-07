#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __Author__ = 'Tannon Kew'
# __Email__ = 'kew@cl.uzh.ch
# __Date__ = '2023-03-03'

import os
import sys
import time
from pathlib import Path
import random
import logging
from typing import List, Dict, Tuple, Optional, Union

from tqdm import tqdm
from transformers import HfArgumentParser, set_seed

from utils import iter_batches, iter_json_lines, serialize_to_jsonl, get_output_file_name, persist_args
from prompt_utils import prepare_prompted_inputs, RandomExampleSelector, postprocess_model_outputs
from llm_inference import InferenceArguments, LLM

from model_utils import setup_model_parallel, load

logger = logging.getLogger(__name__)

def run_inference(args):

    # set random seed everywhere for reproducibility
    set_seed(args.seed)

    if "llama" in args.model_name_or_path.lower(): # special case for Facebook's LLaMA model
        logger.info("Loading LLaMA model")
        local_rank, world_size = setup_model_parallel(args.seed)
        if local_rank > 0:
            sys.stdout = open(os.devnull, 'w')
            
        tokenizer_path = str(Path(args.model_name_or_path).parent / "tokenizer.model")
        llm = load(
            args.model_name_or_path, 
            tokenizer_path, 
            local_rank, 
            world_size,
            max_seq_len=512,
            max_batch_size=args.batch_size
            )
    else:
        llm = LLM(args)
    
    # Use stdout when output_file and output_dir is not specified (e.g. for debugging)
    if not args.output_file and not args.output_dir:
        args.output_file = "stdout"
    elif args.output_file and not args.output_dir:
        args.output_file = Path(args.output_file)
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
    elif not args.output_file and args.output_dir:
        args.output_file = get_output_file_name(args)
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    else:
        raise RuntimeError(f"Could not infer output file!")
    
    if args.output_file != "stdout":
        persist_args(args)
    
    # prepare few-shot examples
    examples = list(iter_json_lines(args.examples))
    logger.info(f"Few-shot examples will be sampled from {len(examples)} items")
    example_selector = RandomExampleSelector(
            examples=examples, # the examples it has available to choose from.
            few_shot_n=args.few_shot_n,
            n_refs=args.n_refs,
        )

    with open(args.output_file, "w", encoding="utf8") if args.output_file != "stdout" else sys.stdout as outf:
        start_time = time.time()
        c = 0 # counter for generated output sequences

        for input_batch in tqdm(iter_batches(args.input_file, args.batch_size)):
            # input file can be a text file or a jsonl file, in the latter case 
            # we assume that the input sentence is in the key specified by args.source_key
            if isinstance(input_batch[0], dict):
                input_batch = [i[args.source_key] for i in input_batch]
            
            inputs = prepare_prompted_inputs(
                inputs=input_batch,
                example_selector=example_selector,
                prefix=args.prompt_prefix,
                suffix=r"Complex: {input}\nSimple:",
                example_separator=args.example_separator,
                prompt_format=args.prompt_format,
            )

            if "llama" in args.model_name_or_path.lower():
                outputs = llm.generate(
                    inputs, 
                    max_gen_len=args.max_new_tokens, 
                    temperature=args.temperature, 
                    top_p=args.top_p
                    )
                # pack outputs into a list of lists to match expected format from HF models
                outputs = [[o] for o in outputs]
            else:
                outputs = llm.generate_from_model(inputs)

            outputs = postprocess_model_outputs(inputs, outputs, args.example_separator)

            for line in serialize_to_jsonl(inputs, outputs):
                outf.write(f"{line}\n")
                c += 1

        end_time = time.time()
        logger.info(f"Finised inference on {args.input_file} in {end_time - start_time:.4f} seconds.")
        logger.info(f"Wrote {c} outputs to {args.output_file}")

if __name__ == '__main__':
    parser = HfArgumentParser((InferenceArguments))
    args = parser.parse_args_into_dataclasses()[0]
    run_inference(args)