#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import logging
from typing import List, Dict, Tuple, Optional, Union
from transformers import HfArgumentParser

from utils import iter_batches
from llm_inference import InferenceArguments, LLM

logger = logging.getLogger(__name__)

def prepare_inputs(examples: List[Dict], few_shot_n: int = 0, delimiter: str = '***', seed: int = 42) -> List[str]:
    """
    prepares few-shot (or zero-shot) inputs for LLM inference
    """
    
    inputs = []
    for ex in examples:
        if few_shot_n == 0:
            inputs.append(ex['src'])
        elif isinstance(ex['examples'], list):
            assert few_shot_n <= len(ex['examples']), f"few_shot_n ({few_shot_n}) can not be greater than the number of available examples ({len(ex['examples'])})"
            input_str = delimiter.join(random.sample(ex['examples'], few_shot_n)) + delimiter + ex['src']
            inputs.append(input_str)
        else:
            raise NotImplementedError(f'Expected examples to be a list of examples, but got {type(ex["examples"])}')
    return inputs

if __name__ == '__main__':

    parser = HfArgumentParser((InferenceArguments))
    args = parser.parse_args_into_dataclasses()[0]

    # set random seed for reproducibility
    random.seed(args.seed)

    llm = LLM(args.model_name_or_path, args.max_memory, args.seed)

    for batch_prompts in iter_batches(args.input_file, args.batch_size):
        inputs = prepare_inputs(batch_prompts, args.few_shot_n, args.delimiter, args.seed)
        outputs = llm.generate_from_model(inputs, args)
    
        outputs = llm.postprocess_model_outputs(inputs, outputs, args.delimiter)
        for o in outputs:
            print(o)