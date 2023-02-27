#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from typing import List, Dict, Iterable, Optional

from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.prompts import load_prompt
from langchain.prompts.example_selector import LengthBasedExampleSelector
from langchain.prompts.example_selector.base import BaseExampleSelector
from langchain.prompts.example_selector.ngram_overlap import NGramOverlapExampleSelector

from utils import iter_json_lines


class RandomExampleSelector(BaseExampleSelector):
    
    def __init__(self, examples: List[Dict[str, str]], few_shot_n: int = 1, n_refs: int = 1, seed: int = 42):
        self.examples = examples
        self.few_shot_n = few_shot_n
        self.n_refs = n_refs
        
        random.seed(seed) # set seed for reproducibility

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
    
simple_prompt = PromptTemplate(
    input_variables=["complex", "simple"],
    template="Complex: {complex}\nSimple: {simple}",
)
    
def prepare_prompted_inputs(
    inputs: List[str],
    examples: List[Dict], 
    prefix: str = "I want you to replace my complex sentence with simple sentence(s). Keep the meaning same, but make them simpler.",
    suffix: str = "Complex: {input}\nSimple:",
    few_shot_n: int = 0, 
    n_refs: int = 1,
    example_separator: str = "\n\n",
    ref_delimiter: str = "\t",
    seed: int = 42,
    ):

    prompted_inputs = []
    for i in inputs:
        example_selector = RandomExampleSelector(
            examples=examples, # the examples it has available to choose from.
            few_shot_n=few_shot_n,
            n_refs=n_refs,
            seed=seed,
        )

        few_shot_prompt = FewShotPromptTemplate(
            example_selector=example_selector, # use an ExampleSelector instead of examples.
            example_prompt=simple_prompt, # examples format
            prefix=prefix,
            suffix=suffix,
            input_variables=["input"], # the variables that the overall prompt expects
            example_separator=example_separator, # string used to join the prefix, examples, and suffix together
        )

        prompted_inputs.append(few_shot_prompt.format(input=i))

    return prompted_inputs

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
    

if __name__ == "__main__":

    # load_prompt("prompts/ss_p1.json") # doesn't allow for example selector (?)

    dataset = "data/asset/dataset/valid.jsonl"
    n_refs = 2
    seed = 3

    print(prepare_prompted_inputs(
        inputs = ["It is particularly famous for the cultivation of kiwifruit."],
        examples = list(iter_json_lines(dataset)),
        few_shot_n = 3,
        n_refs = n_refs,
        seed=10902
    ))
