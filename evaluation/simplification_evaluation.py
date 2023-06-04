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
        resources/outputs/bloom-560m/asset-test_asset-valid_p0_fs3_nr1_s489.jsonl \
        --use_cuda \
        --lens_model_path resources/LENS/checkpoints/epoch=5-step=6102.ckpt \
        --out_file resources/outputs/bloom-560m/asset-test_asset-valid_p0_fs3_nr1_s489.eval

optional:
    --src_file e.g. resources/data/asset/dataset/asset.test.jsonl
    --ref_file e.g. resources/data/asset/dataset/asset.test.simp.4
    --use_cuda
    --lens_model_path e.g. resources/LENS/checkpoints/epoch=5-step=6102.ckpt

"""

import os
import random
import logging
import argparse
from typing import List, Optional, Iterable

import numpy as np
from tqdm import tqdm
import pandas as pd
import torch # for bertscore
from easse import sari, bleu, fkgl, bertscore, quality_estimation # samsa fails dep: tupa
import lens
from lens.lens_score import LENS # https://github.com/Yao-Dou/LENS/tree/master/lens

from evaluation.distinct_n import distinct
from evaluation.perplexity import score_ppl
from utils.helpers import iter_lines

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # hack required for LENS 

logger = logging.getLogger(__name__)

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('hyp_file', type=str, help='Path to a JSONL file with model outputs and optionally source and reference sentences.')
    parser.add_argument('--src_file', type=str, required=False, help='Path to a JSONL file with source and optionally reference sentences.')
    parser.add_argument('--ref_file', type=str, required=False, help='Path to a TXT file with reference sentences. WARING: assumes only one reference set.')
    parser.add_argument('--out_file', type=str, required=False, help='Path to a CSV file with metric scores.')
    parser.add_argument('--use_cuda', action='store_true', help='if provided, model-based metrics such as PPL and BERTScore will be computed on GPU.')
    parser.add_argument('--lens_model_path', type=str, required=False, default='resources/LENS/checkpoints/epoch=5-step=6102.ckpt', help='Path to a LENS model (see https://github.com/Yao-Dou/LENS/tree/master/lens)')
    return parser.parse_args()

def normalise(scores: Iterable[float]) -> List[float]:
    """Normalise scores to sum to 1."""
    return list(map(lambda x: x * 100, scores))

def compute_metrics(
    src_sents: List[str], 
    ref_sents: List[List[str]], 
    hyp_sents: List[str],
    use_cuda: bool = False,
    lens_model_path: Optional[str] = None,
    ):
    
    results = {}
    
    # basic simplification metrics
    results['bleu'] = bleu.corpus_bleu(hyp_sents, refs_sents)
    
    # results['sari'] = sari.corpus_sari(src_sents, hyp_sents, refs_sents, legacy=False)
    
    # individual SARI scores
    sari_add, sari_keep, sari_del = sari.get_corpus_sari_operation_scores(src_sents, hyp_sents, refs_sents)
    results['sari_add'] = sari_add
    results['sari_keep'] = sari_keep
    results['sari_del'] = sari_del
    results['sari'] = (sari_add + sari_keep + sari_del) / 3

    results['fkgl'] = fkgl.corpus_fkgl(hyp_sents)
    
    results['pbert_ref'], results['rbert_ref'], results['fbert_ref'] = None, None, None
    results['pbert_src'], results['rbert_src'], results['fbert_src'] = None, None, None
    results['ppl_mean'], results['ppl_std'] = None, None
    results['lens'], results['lens_std'] = None, None

    if not torch.cuda.is_available() or not use_cuda:
        logger.warning('Cuda not in use. Skipping PPL and BERTScore and LENS!')
    else:
        logger.info('Using cuda to compute PPL')
        
        results['ppl_mean'], results['ppl_std'] = score_ppl(hyp_sents)

        logger.info('Using cuda to compute BERTScore...')
        # the easse bertscore implementation expext refs_sents to be a 2D list with shape [# ref sets, # refs]
        results['pbert_ref'], results['rbert_ref'], results['fbert_ref'] = normalise(bertscore.corpus_bertscore(hyp_sents, refs_sents))
        results['pbert_src'], results['rbert_src'], results['fbert_src'] = normalise(bertscore.corpus_bertscore(hyp_sents, [src_sents]))

        # lens
        if lens_model_path:
            try:
                lens_metric = LENS(lens_model_path, rescale=True)
                # LENS expects refs to be shape [# samples, # refs]
                refs_sents_ = list(map(list, zip(*refs_sents)))

                lens_scores = lens_metric.score(src_sents, hyp_sents, refs_sents_, batch_size=16, gpus=1)
                results['lens'] = np.mean(lens_scores)
                results['lens_std'] = np.std(lens_scores)
            except Exception as e:
                logger.warning(f'LENS failed with error: {e}')
                
    # distinct    
    results['intra_dist1'], results['intra_dist2'], results['inter_dist1'], results['inter_dist2'] = normalise(distinct(hyp_sents))

    # quality estimation
    qe = quality_estimation.corpus_quality_estimation(src_sents, hyp_sents)
    results.update(qe)

    return results

def load_data(args):
    """
    Load data from files.
    
    Input files may be in jsonl or txt/tsv format.
    """
    # model outputs can be in jsonl or txt format
    if args.hyp_file.endswith('.txt') or args.hyp_file.endswith('.tsv'):
        hyp_sents = list(iter_lines(args.hyp_file))
        hyp_sents = [i.split('\t')[0] for i in hyp_sents] # if tab-separated, take only the first column (assumes that mulitple hyps were returned)
    elif args.hyp_file.endswith('.jsonl'):
        lines = list(iter_lines(args.hyp_file))
        hyp_sents = [i['model_output'] for i in lines]
    
    # simplest case: if src and ref files are provided, load them
    if args.src_file and args.ref_file:
        logger.info(f'Loading src_file {args.src_file}')
        src_sents = list(iter_lines(args.src_file))
        logger.info(f'Loading ref_file {args.ref_file}')
        refs_sents = [list(iter_lines(args.ref_file))]
    
    # if only src file is provided, assume that human refs are also int the src file
    elif args.src_file and not args.ref_file:
        logger.info(f'No ref_file provided. Assuming that src and refs are in src_file {args.src_file}')
        lines = list(iter_lines(args.src_file))
        src_sents = [i['complex'] for i in lines]
        refs_sents = [i['simple'] for i in lines]
    
    # otherwise, if no src file and no ref file are provided, assume src and human refs are in hyp file (jsonl format)
    elif not args.src_file and not args.ref_file:
        logger.info(f'No src_file or ref_file provided. Assuming that src and refs are in hyp_file {args.hyp_file}')
        lines = list(iter_lines(args.hyp_file))
        src_sents = [i['source'] for i in lines]
        refs_sents = [i['references'] for i in lines]

    refs_sents = list(map(list, [*zip(*refs_sents)])) # transpose from [# samples, # refs_per_sample] to [# refs_per_sample, # samples]
    
    # sanity checks
    if len(src_sents) != len(refs_sents[0]):
        raise ValueError('Number of source sentences does not match number of reference sentences')
    if len(src_sents) != len(hyp_sents):
        raise ValueError('Number of source sentences does not match number of hypothesis sentences')

    return src_sents, refs_sents, hyp_sents

if __name__ == '__main__':

    args = set_args()
    
    src_sents, refs_sents, hyp_sents = load_data(args)

    results = compute_metrics(src_sents, refs_sents, hyp_sents, use_cuda=args.use_cuda, lens_model_path=args.lens_model_path)

    # add file id
    results['file_id'] = args.hyp_file
    
    df = pd.DataFrame(data=results, index=[0]).round(4)
    if args.out_file:
        df.to_csv(args.out_file, sep=';', index=False, float_format='%.4f', encoding='utf-8')
        logger.info(f'Wrote results to {args.out_file}')
    # also print to stdout
    print(df.to_csv(sep=';', index=False))