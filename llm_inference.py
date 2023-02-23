#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import math
import time
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

import torch
from transformers import (
    pipeline,
    AutoModelForCausalLM, 
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512mb"


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

@dataclass
class InferenceArguments:
    """
    Arguments pertaining to running generation/inference with pre-trained/fine-tuned model.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )    

    # checkpoint_dir: str = field(
    #     default=None,
    #     metadata={"help": "Path to fine-tuned model checkpoint"}
    # )
    
    output_dir: str = field(
        default=None,
        metadata={"help": "Path to output directory"}
    )

    output_file: str = field(
        default=None,
        metadata={"help": "Output file for model generations"}
    )

    input_file: str = field(
        default=None,
        metadata={"help": "Input file containing prompt generations"}
    )

    seed: int = field(
        default=42,
        metadata={"help": "random seed"}
    )

    use_cuda: bool = field(
        default=True,
        metadata={"help": "Use GPU if available"}
    )

    batch_size: int = field(
        default=1,
        metadata={"help": "Batch size for predictions"}
    )

    min_length: int = field(
        default=None,
        metadata={"help": "Minimum length of generated text"}
    )

    max_length: int = field(
        default=64,
        metadata={"help": "Maximum length of generated text"}
    )

    max_new_tokens: int = field(
        default=100,
        metadata={"help": "Maximum number of tokens to generate"}
    )

    length_penalty: float = field(
        default=1.0,
        metadata={"help": "Length penalty for generated text"}
    )

    no_early_stop: bool = field(
        default=False,
        metadata={"help": "Disable early stopping on generate"}
    )

    num_return_sequences: int = field(
        default=1,
        metadata={"help": "Number of sequences to generate"}
    )

    num_beams: int = field(
        default=4,
        metadata={"help": "Number of beams for beam search"}
    )

    do_sample: bool = field(
        default=False,
        metadata={"help": "Sample instead of greedy decoding"}
    )

    temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for generation"}
    )
    
    top_k: int = field(
        default=0,
        metadata={"help": "Number of top k tokens to keep for top-k sampling"}
    )

    top_p: float = field(
        default=0.0,
        metadata={"help": "Probability of top-p sampling"}
    )

    # write_to_file: str = field(
    #     default='auto',
    #     metadata={"help": "Output file for generated text or `auto` to generate outfile name based on generation parameters"}
    # )

    verbose: bool = field(
        default=False,
        metadata={"help": "Print progress"}
    )

    data_seed: int = field(
        default=42,
        metadata={"help": "random seed for data loading"}
    )

    debug: bool = field(
        default=False,
        metadata={"help": "Print debug information"}
    )

    prompt: str = field(
        default=None,
        metadata={"help": "Prompt for generated text"}
    )

    # prompts: List[str] = field(
    #     default=None,
    #     metadata={"help": "Prompt for generated text"}
    # )

    few_shot_n: int = field(
        default=0,
        metadata={"help": "number of examples to use as few-shot in-context examples"}
    )

    delimiter: str = field(
        default="\n\n",
        metadata={"help": "Delimiter for prompts and generated text"}
    )

    max_memory: float = field(
        default=1.0,
        metadata={"help": "Prompt for generated text"}
    )


class LLM(object):

    def __init__(self, model_name: str, max_memory: Optional[int] = None):
        # https://github.com/huggingface/accelerate/issues/864#issuecomment-1327726388    
        start_time = time.time()
        # balanced_low_0 is useful for when you need to use GPU 0 for some processing of the outputs, e.g. when using the generate function
        # breakpoint()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            # device_map="balanced_low_0", 
            device_map="auto", 
            load_in_8bit=True, 
            torch_dtype=torch.float16, 
            max_memory=self.set_max_memory(max_memory),
            offload_state_dict=True,
            offload_folder="/scratch/tkew/offload" # TODO: make this configurable
            )
        end_time = time.time()
        logger.info(f"Loaded model {model_name} in {end_time - start_time:.4f} seconds")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"Model footprint {self.model.get_memory_footprint() / (1024*1024*1024):.4f} GB")

    @staticmethod
    def set_max_memory(max_memory: Optional[float] = None):
        if max_memory and max_memory != 1.0:
            n_gpus = torch.cuda.device_count()
            logger.info(f"Infering max memory...")
            t = torch.cuda.get_device_properties(0).total_memory / (1024*1024*1024)
            # note, we user math.floor() as a consertative rounding method
            # to optimize the maximum batch size on multiple GPUs, we give the first GPU less memory
            # see max_memory at https://huggingface.co/docs/accelerate/main/en/usage_guides/big_modeling
            max_memory = {
                i:(f"{math.floor(t*max_memory)}GiB" if i > 0 else f"{math.floor(t*max_memory*0.6)}GiB") for i in range(n_gpus)
                }
            max_memory['cpu'] = '400GiB'
            
            logger.info(f"Set maximum memory: {max_memory}")
            return max_memory
        else:
            return None

    def generate_from_model(self, inputs: List[str], args: InferenceArguments) -> List[str]:
        """
        queries the generation model for a given batch of inputs
        """
        encoded_inputs = self.tokenizer(inputs, return_tensors='pt', padding=True)
        start_time = time.time()
        model_outputs = self.model.generate(
            input_ids=encoded_inputs['input_ids'].cuda(), 
            max_new_tokens=args.max_new_tokens, 
            min_length=args.min_length,
            num_beams=args.num_beams,
            num_return_sequences=args.num_return_sequences, 
            early_stopping=not args.no_early_stop,
            do_sample=args.do_sample, 
            temperature=args.temperature, 
            top_k=args.top_k, 
            top_p=args.top_p,
            )#.to('cpu')
        end_time = time.time()

        if args.verbose:
            logger.info(f"Generated {sum(model_outputs.shape) - sum(encoded_inputs['input_ids'].shape)} tokens in {end_time - start_time:.4f} seconds")

        # self.tokenizer.batch_decode(model_outputs[:, encoded_inputs['input_ids'].size()[1]:], skip_special_tokens=True)
        return self.tokenizer.batch_decode(model_outputs, skip_special_tokens=True)

    @staticmethod
    def postprocess_model_outputs(inputs: List[str], outputs: List[str], delimiter: str = '***') -> List[str]:
        trimmed_outputs = []
        for i, o in zip(inputs, outputs):
            o = o.replace(i, '').strip() # remove the input substring (prompt) from the output string
            o = o.split(delimiter) # e.g. '\\n\\n' if used as prompt delimiter and to allow cuting off after the first example
            if len(o) == 1:
                logger.warning(f"Delimiter '{delimiter}' not found in output {o[:50]}...")
            trimmed_outputs.append(o[0].strip())
        return trimmed_outputs


if __name__ == "__main__":
    
    hf_parser = HfArgumentParser((InferenceArguments))
    args = hf_parser.parse_args_into_dataclasses()[0]

    llm = LLM(args.model_name_or_path, args.max_memory)

    # print(llm.generate_from_model(args.prompt, max_new_tokens=args.max_new_tokens, verbose=args.verbose))
    print(llm.generate_from_model([args.prompt], args))
