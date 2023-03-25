#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# __Author__ = 'Tannon Kew'
# __Email__ = 'kew@cl.uzh.ch
# __Date__ = '2023-03-16'

"""

Computes simplification metrics for a given file of hypotheses. 
If no source and reference files are provided, assumes source and reference sentences are in the hypothesis file.

Example call:

    python -m evaluation.simplification_evaluation \
        data/outputs/bloom-560m/asset-test-head12_asset-valid_69e6d564-prefix-initial_3_1_42.jsonl

optional:
    --src_file e.g. data/asset/dataset/asset.test.jsonl
    --ref_file e.g. data/asset/dataset/asset.test.simp.4
    --use_cuda

"""

import random
import logging
import argparse
from typing import List, Optional, Iterable

import numpy as np
from tqdm import tqdm
import pandas as pd
import torch # for bertscore
from easse import sari, bleu, fkgl, bertscore, quality_estimation # samsa fails dep: tupa

from evaluation.distinct_n import distinct
from evaluation.perplexity import score_ppl
from utils.helpers import iter_lines

logger = logging.getLogger(__name__)

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('hyp_file', type=str, help='Path to a JSONL file with model outputs and optionally source and reference sentences.')
    parser.add_argument('--src_file', type=str, required=False, help='Path to a JSONL file with source and optionally reference sentences.')
    parser.add_argument('--ref_file', type=str, required=False, help='Path to a TXT file with reference sentences. WARING: assumes only one reference set.')
    parser.add_argument('--out_file', type=str, required=False, help='Path to a CSV file with metric scores.')
    parser.add_argument('--use_cuda', action='store_true', help='if provided, model-based metrics such as PPL and BERTScore will be computed on GPU.')
    return parser.parse_args()

def normalise(scores: Iterable[float]) -> List[float]:
    """Normalise scores to sum to 1."""
    return list(map(lambda x: x * 100, scores))

def compute_metrics(
    src_sents: List[str], 
    ref_sents: List[List[str]], 
    hyp_sents: List[str],
    use_cuda: bool = False,
    ):
    
    results = {}
    
    # basic simplification metrics
    results['bleu'] = bleu.corpus_bleu(hyp_sents, refs_sents)
    results['sari'] = sari.corpus_sari(src_sents, hyp_sents, refs_sents, legacy=False)
    results['fkgl'] = fkgl.corpus_fkgl(hyp_sents)
    
    results['pbert_ref'], results['rbert_ref'], results['fbert_ref'] = None, None, None
    results['pbert_src'], results['rbert_src'], results['fbert_src'] = None, None, None
    results['ppl_mean'], results['ppl_std'] = None, None

    if not torch.cuda.is_available() or not use_cuda:
        logger.warning('Cuda not in use. Skipping PPL and BERTScore!')
    else:
        logger.info('Using cuda to compute PPL')
        
        results['ppl_mean'], results['ppl_std'] = score_ppl(hyp_sents)

        logger.info('Using cuda to compute BERTScore...')
        # the easse bertscore implementation expext refs_sents to be a 2D list with shape [# ref sets, # refs]
        results['pbert_ref'], results['rbert_ref'], results['fbert_ref'] = normalise(bertscore.corpus_bertscore(hyp_sents, refs_sents))
        results['pbert_src'], results['rbert_src'], results['fbert_src'] = normalise(bertscore.corpus_bertscore(hyp_sents, [src_sents]))

    # distinct    
    results['intra_dist1'], results['intra_dist2'], results['inter_dist1'], results['inter_dist2'] = normalise(distinct(hyp_sents))

    # quality estimation
    qe = quality_estimation.corpus_quality_estimation(src_sents, hyp_sents)
    results.update(qe)

    return results

if __name__ == '__main__':

    args = set_args()

    lines = list(iter_lines(args.hyp_file))

    hyp_sents = [i['model_output'] for i in lines]

    if not args.src_file and not args.ref_file: # assumes src and human refs are in hyp file
        src_sents = [i['source'] for i in lines]
        refs_sents = [i['references'] for i in lines]
        refs_sents = list(map(list, [*zip(*refs_sents)])) # transpose from [# samples, # refs] to [# refs, # samples]
    elif args.src_file and not args.ref_file: # assumes human refs are in src file
        lines = list(iter_lines(args.src_file))
        src_sents = [i['complex'] for i in lines]
        refs_sents = [i['simple'] for i in lines]
        refs_sents = list(map(list, [*zip(*refs_sents)])) # transpose from [# samples, # refs] to [# refs, # samples]
    elif args.src_file and args.ref_file: # assumes single set of human refs are in ref file
        src_sents = list(iter_lines(args.src_file))
        refs_sents = [list(iter_lines(args.ref_file))]
    
    # sanity checks
    if len(src_sents) != len(refs_sents[0]):
        raise ValueError('Number of source sentences does not match number of reference sentences')
    if len(src_sents) != len(hyp_sents):
        raise ValueError('Number of source sentences does not match number of hypothesis sentences')

    results = compute_metrics(src_sents, refs_sents, hyp_sents, use_cuda=args.use_cuda)

    # add file id
    results['file_id'] = args.hyp_file
    
    df = pd.DataFrame(data=results, index=[0]).round(4)
    if args.out_file:
        df.to_csv(args.out_file, sep=';', index=False, float_format='%.4f', encoding='utf-8')
        logger.info(f'Wrote results to {args.out_file}')
    # also print to stdout
    print(df.to_csv(sep=';', index=False))