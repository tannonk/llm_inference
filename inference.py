#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import logging
from typing import List, Dict, Tuple, Optional, Union
from transformers import HfArgumentParser

from utils import iter_batches, iter_json_lines
from prompt_utils import prepare_prompted_inputs
from llm_inference import InferenceArguments, LLM

logger = logging.getLogger(__name__)


if __name__ == '__main__':

    parser = HfArgumentParser((InferenceArguments))
    args = parser.parse_args_into_dataclasses()[0]

    # set random seed for reproducibility
    random.seed(args.seed)

    llm = LLM(args.model_name_or_path, args.max_memory, args.seed)

    examples = list(iter_json_lines(args.examples))

    for input_batch in iter_batches(args.input_file, args.batch_size):
        inputs = prepare_prompted_inputs(
            inputs=input_batch,
            examples=examples,
            prefix=args.prompt_prefix,
            suffix="Complex: {input}\nSimple:",
            few_shot_n=args.few_shot_n,
            n_refs=args.n_refs,
            example_separator=args.example_separator,
            # ref_delimiter=args.ref_delimiter,
            seed=args.seed,   
        )

        # inputs = prepare_inputs(batch_prompts, args.few_shot_n, args.delimiter, args.seed)
        outputs = llm.generate_from_model(inputs, args)
    
        outputs = llm.postprocess_model_outputs(inputs, outputs, args.example_separator)
        for o in outputs:
            print(o)