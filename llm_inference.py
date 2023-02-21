#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
        default=3,
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

    prompts: List[str] = field(
        default=None,
        metadata={"help": "Prompt for generated text"}
    )

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
        max_memory = self.set_max_memory(max_memory)
        
        start_time = time.time()
        # balanced_low_0 is useful for when you need to use GPU 0 for some processing of the outputs, e.g. when using the generate function
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="balanced_low_0", 
            load_in_8bit=True, 
            torch_dtype=torch.float16, 
            max_memory=max_memory,
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
                i:(f"{math.floor(t*max_memory)}GB" if i > 0 else f"{math.floor(t*max_memory*0.5)}GB") for i in range(n_gpus)
                }
            
            logger.info(f"Set maximum memory: {max_memory}")
        else:
            return None


    # def generate_from_model(self, prompt: str, args):
    #     encoded_input = self.tokenizer(prompt, return_tensors='pt')
    #     start_time = time.time()
    #     model_outputs = self.model.generate(input_ids=encoded_input['input_ids'].cuda(), max_new_tokens=args.max_new_tokens)
    #     end_time = time.time()
    #     if args.verbose:
    #         logger.info(f"Generated {sum(model_outputs.shape) - sum(encoded_input['input_ids'].shape)} tokens in {end_time - start_time:.4f} seconds")
    #     return self.tokenizer.decode(model_outputs[0], skip_special_tokens=True)


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
        
        # breakpoint()
        # self.tokenizer.batch_decode(model_outputs[:, encoded_inputs['input_ids'].size()[1]:], skip_special_tokens=True)
        return self.tokenizer.batch_decode(model_outputs, skip_special_tokens=True)

    @staticmethod
    def postprocess_model_outputs(inputs: List[str], outputs: List[str], delimiter: str = '***') -> List[str]:
        trimmed_outputs = []
        # breakpoint()
        for i, o in zip(inputs, outputs):
            o = o.replace(i, '').strip() # remove the input substring (prompt) from the output string
            o = o.split(delimiter) # e.g. '\\n\\n' if used as prompt delimiter and to allow cuting off after the first example
            if len(o) == 1:
                logger.warning(f"Delimiter '{delimiter}' not found in output {o[:50]}...")
            trimmed_outputs.append(o[0].strip())
        return trimmed_outputs


    # def batch_for_generation(self, examples, batch_size: int):
        
    #     current_batch = {
    #         'input_ids': [], 
    #         'attention_mask': [], 
    #         'labels': [],
    #     }

    #     for i, example in enumerate(examples):
    #         current_batch['input_ids'].append(example['input_ids'])
    #         current_batch['attention_mask'].append(example['attention_mask'])
    #         current_batch['labels'].append(example['labels'])
    #         current_batch['turns'].append(example.get('turns'))
    #         current_batch['knowledge'].append(example.get('knowledge'))
    #         current_batch['target'].append(example.get('target'))

    #         # get cross attention biases for each individual example (would be fast to do for a batch if all items are the same)
    #         # if self.gen_args.use_cross_attention_bias:
    #         current_batch['cross_attention_bias'].append(self.construct_cross_attention_bias(example['attention_mask']))
    #         if context_code is not None:
    #             current_batch['context_code'].append(context_code)

    #         if len(current_batch['input_ids']) == batch_size or i == len(examples) - 1:
                
    #             current_batch['input_ids'] = torch.stack(current_batch['input_ids']).to(self.model.device)
    #             current_batch['attention_mask'] = torch.stack(current_batch['attention_mask']).to(self.model.device)
    #             # TODO: pad to max length before stacking to return true tensor
    #             # current_batch['labels'] = torch.stack(current_batch['labels']).to(model.device)
                
    #             # stack cross attention biases for each individual example
    #             current_batch['cross_attention_bias'] = torch.stack(current_batch['cross_attention_bias']).to(self.model.device) if len(current_batch['cross_attention_bias']) else None
    #             # to make sure dimensions are correct, we need to ensure the attention 
    #             # bias vector as the same length as the encoder hidden states which differs depending on the context code
    #             current_batch['cross_attention_bias'] = self.expand_attention_bias_to_context_code(current_batch['cross_attention_bias'], context_code)
    #             # stack context code for each individual example
    #             current_batch['context_code'] = torch.stack(current_batch['context_code']) if len(current_batch['context_code']) else None

    #             yield current_batch
            
    #             # reset lists for next batch
    #             current_batch['input_ids'], current_batch['attention_mask'], current_batch['labels'] = [], [], []
    #             current_batch['turns'], current_batch['knowledge'], current_batch['target'] = [], [], []
    #             current_batch['cross_attention_bias'] = []
    #             current_batch['context_code'] = []



if __name__ == "__main__":
    
    hf_parser = HfArgumentParser((InferenceArguments))
    args = hf_parser.parse_args_into_dataclasses()[0]

    llm = LLM(args.model_name_or_path, args.max_memory)

    # print(llm.generate_from_model(args.prompt, max_new_tokens=args.max_new_tokens, verbose=args.verbose))
    print(llm.generate_from_model([args.prompt], args))
