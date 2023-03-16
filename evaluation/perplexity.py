#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __Author__ = 'Tannon Kew'
# __Email__ = 'kew@cl.uzh.ch
# __Date__ = '2023-03-16'

from typing import List, Tuple
import os
import numpy as np
import evaluate # https://huggingface.co/docs/evaluate/a_quick_tour

os.environ['TOKENIZERS_PARALLELISM']='false'

perplexity = evaluate.load("perplexity", module_type="metric")

def score_ppl(input_texts: List[str], model_id: str = 'distilgpt2', batch_size: int = 32,) -> Tuple[float, float]:
    """
    uses https://huggingface.co/spaces/evaluate-metric/perplexity
    By default, model will use cuda if available!
    print(list(results.keys()))
    >>>['perplexities', 'mean_perplexity']
    """
    ppl = perplexity.compute(model_id='distilgpt2', add_start_token=True, predictions=input_texts, batch_size=batch_size)
    
    return ppl["mean_perplexity"], np.array(ppl['perplexities']).std()

if __name__ == '__main__':
    pass