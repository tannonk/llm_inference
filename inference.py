#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

logger = logging.getLogger(__name__)

def run_inference(args):

    # set random seed everywhere for reproducibility
    set_seed(args.seed)
    
    llm = LLM(args)
    
    # Use stdout when output_file and output_dir is not specified (e.g. for debugging)
    if not args.output_file and not args.output_dir:
        args.output_file = "stdout"
    elif not args.output_file:
        args.output_file = Path(args.output_file)
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
    elif not args.output_dir:
        args.output_file = get_output_file_name(args)
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
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
            )

            print(inputs)
            breakpoint()
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