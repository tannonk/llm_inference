#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import random
import logging
from typing import List, Dict, Iterable, Optional

from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.prompts import load_prompt
from langchain.prompts.example_selector import LengthBasedExampleSelector
from langchain.prompts.example_selector.base import BaseExampleSelector
from langchain.prompts.example_selector.ngram_overlap import NGramOverlapExampleSelector

from utils import iter_json_lines

logger = logging.getLogger(__name__)

simple_prompt = PromptTemplate(
    input_variables=["complex", "simple"],
    template=r"Complex: {complex}\nSimple: {simple}",
)

class RandomExampleSelector(BaseExampleSelector):
    
    def __init__(
        self, 
        examples: List[Dict[str, str]], 
        few_shot_n: int = 1, 
        n_refs: int = 1, 
        ):
        self.examples = examples
        self.few_shot_n = few_shot_n
        self.n_refs = n_refs
        
    def add_example(self, example: Dict[str, str]) -> None:
        """Add new example to store for a key."""
        self.examples.append(example)

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Select which examples to use based on the inputs.""" 
        return self.flatten_references(random.sample(self.examples, self.few_shot_n), n_refs=self.n_refs)

    @staticmethod
    def flatten_references(
        examples: Iterable, 
        src_key: str = "complex", 
        tgt_key: str = "simple", 
        n_refs: int = 1, 
    ):
        """
        Handles multi-reference examples for few-shot prompting.

        Datasets such as ASSET provide 10 reference simplifications for each complex sentence.
        We select n_refs at random from the available reference simplifications to provide as examples in few-shot prompting.
        Each reference simplification is separated by '\\t'

        Args:
            examples :: an iterable containing dictionaries with src_key and tgt_key
            src_key :: name of the key for the src sequence
            tgt_key :: name of the key for the tgt sequence
            n_refs :: number of target references to use in a prompt

        """
        flat_examples = []
        for ex in examples:
            flat_ex = {}
            flat_ex["complex"] = ex[src_key]
            if isinstance(ex[tgt_key], list):
                simple_references = random.sample(ex[tgt_key], n_refs)
                simple_references = [f"{i}: {ref}" for i, ref in enumerate(simple_references)]
                flat_ex["simple"] = f' '.join(simple_references)
            else:
                flat_ex["simple"] = ex[tgt_key]
            flat_examples.append(flat_ex) 
        return flat_examples


def prepare_prompted_inputs(
    inputs: List[str],
    examples: Optional[List[Dict]] = None, 
    example_selector: Optional[BaseExampleSelector] = None,
    prefix: str = "I want you to replace my complex sentence with simple sentence(s). Keep the meaning same, but make them simpler.",
    suffix: str = "Complex: {input}\nSimple:",
    few_shot_n: int = 0, 
    n_refs: int = 1,
    example_separator: str = r"\n\n",
    ref_delimiter: str = r"\t",
    ):

    if examples is None and example_selector is None:
        raise RuntimeError(f"Expected either `examples` or a valid `example_selector` but got None")

    prompted_inputs = []

    for i in inputs:
        
        few_shot_prompt = FewShotPromptTemplate(
            examples=None if example_selector is None else examples,
            example_selector=example_selector, # use an ExampleSelector instead of examples.
            example_prompt=simple_prompt, # examples format
            prefix=prefix,
            suffix=suffix,
            input_variables=["input"], # the variables that the overall prompt expects
            example_separator=example_separator, # string used to join the prefix, examples, and suffix together
        )

        prompted_inputs.append(few_shot_prompt.format(input=i))

    return prompted_inputs

# def flatten_references(examples: Iterable, src_key: str = "complex", tgt_key: str = "simple", n_refs: int = 1) -> List[Dict]:
#     """
#     Handles multi-reference examples for few-shot prompting.

#     Datasets such as ASSET provide 10 reference simplifications for each complex sentence.
#     We select n_refs at random from the available reference simplifications to provide as examples in few-shot prompting.
#     Each reference simplification is separated by '\\t'

#     Args:
#         examples :: an iterable containing dictionaries with src_key and tgt_key
#         src_key :: name of the key for the src sequence
#         tgt_key :: name of the key for the tgt sequence
#         n_refs :: number of target references to use in a prompt

#     """
#     flat_examples = []
#     for ex in examples:
#         flat_ex = {}
#         flat_ex["complex"] = ex[src_key]
#         if isinstance(ex[tgt_key], list):
#             flat_ex["simple"] = '\t'.join(random.sample(ex[tgt_key], n_refs))
#         else:
#             flat_ex["simple"] = ex[tgt_key]
#         flat_examples.append(flat_ex) 
#     return flat_examples

# Alternative input expected - currently not used
# def flatten_references(examples, src_key: str = "complex", tgt_key: Optional[str] = None, n_refs: int = 1):
#     """
#     Handles multi-reference examples for few-shot prompting 

#     Datasets such as ASSET provide 10 reference simplifications for each complex sentence.
#     We select n_refs at random from the available reference simplifications to provide as examples in few-shot prompting.
#     Each reference simplification is separated by '\\t'
#     """
#     flat_examples = [{} for _ in examples]

#     for i in range(len(examples)):
#         flat_examples[i]["complex"] = examples[i].pop(src_key) # removes the source key,value pair
#         if tgt_key is None:
#             tgt_keys = random.sample(list(examples[i].keys()), n_refs) # assumes all remaining keys are potential targets
#             flat_examples[i]["simple"] = '\t'.join([examples[i].get(tgt_key) for tgt_key in tgt_keys]) # randomly select references for example
#         else:
#             flat_examples[i]["simple"] = examples[i].pop(tgt_key)
#     return flat_examples


# redundant
# def prepare_inputs(examples: List[Dict], few_shot_n: int = 0, delimiter: str = '***', seed: int = 42) -> List[str]:
#     """
#     prepares few-shot (or zero-shot) inputs for LLM inference
#     """
    
#     inputs = []
#     for ex in examples:
#         if few_shot_n == 0:
#             inputs.append(ex['src'])
#         elif isinstance(ex['examples'], list):
#             assert few_shot_n <= len(ex['examples']), f"few_shot_n ({few_shot_n}) can not be greater than the number of available examples ({len(ex['examples'])})"
#             input_str = delimiter.join(random.sample(ex['examples'], few_shot_n)) + delimiter + ex['src']
#             inputs.append(input_str)
#         else:
#             raise NotImplementedError(f'Expected examples to be a list of examples, but got {type(ex["examples"])}')
#     return inputs

def postprocess_model_outputs(inputs: List[str], outputs: List[List[str]], example_separator: str = r'\n\n', ref_delimiter: str = r'\t') -> List[List[str]]:
    """
    Applies task-specific post-processing to model output sequences:
        - removes the input sequence
        - trims each output sequence according to the context delimiter provided (i.e. takes only the first one)
    
    Args:
        inputs :: List of prompted input (used to clean up output sequences)
        outputs :: 2D list with shape [inputs, num_return_sequences]
        example_separator :: character delimiter used to seperated few-shot examples
        ref_delimiter :: character delimiter used to seperate multiple reference examples if available (ignored for now)
    """
    
    # initialise a list of lists for outputs in case num_return_sequences > 1
    trimmed_outputs = [[] for _ in range(len(outputs))]
    
    for i in range(len(trimmed_outputs)):
        for out_seq in outputs[i]:
            # out_seq contains the full prompt + generated output
            
            # step 1. strip away the input prompt
            out_seq = out_seq.replace(inputs[i], '').strip() # remove the input substring (prompt) from the output string

            # step 2. split output string by the example seperator character and take only the first part
            split_out_seq = out_seq.split(example_separator) # e.g. '\n\n' if used as example_separator in prompt and to allow cuting off after the first example
            # split_out_seq = re.split(example_separator, out_seq, 1)

            if len(split_out_seq) == 1:
                logger.warning(
                    f"Potentially unfinished sequence " \
                    f"(Delimiter '{example_separator}' not found in output sequence) " \
                    f"You may need to increase `--max_new_tokens` for this task."
                    )                

            # step 3. remove extraneous newlines chars
            out_seq = re.sub(r'\n', ' ', split_out_seq[0])

            # step 4. if multiple references are provided for each prompt example, the model may replicate this pattern
            # currently assumes multiple references are simply enumerated, e.g. 0: ... 1: ...
            out_seq = [x.strip() for x in re.split(r'\d:', out_seq) if x.strip()][0]

            # step 5. remove extraenous task-specific delims            
            out_seq = re.sub(r'\s+Simple:\s+', '', out_seq)
            out_seq = re.sub(r'\s+Complex:\s+', '', out_seq)

            trimmed_outputs[i].append(out_seq.strip())
    return trimmed_outputs  

if __name__ == "__main__":

    # load_prompt("prompts/ss_p1.json") # doesn't allow for example selector (?)

    dataset = "data/asset/dataset/valid.jsonl"
    n_refs = 2
    # seed = 3

    print(prepare_prompted_inputs(
        inputs = ["It is particularly famous for the cultivation of kiwifruit."],
        examples = list(iter_json_lines(dataset)),
        few_shot_n = 3,
        n_refs = n_refs,
        # seed=seed
    ))
