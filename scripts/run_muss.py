#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __Author__ = 'Sweta Agrawal'
# __Email__ = 'sweagraw@umd.edu'
# __Date__ = '2023-05-26'

"""

This script generates MUSS outputs for various datasets. Adapted from https://github.com/facebookresearch/muss/blob/main/muss/simplify.py.

"""


import argparse
import shutil
import re
import time
import logging

from muss.preprocessors import get_preprocessors, ComposedPreprocessor
from muss.utils.helpers import write_lines, read_lines, get_temp_filepath
from muss.fairseq.base import fairseq_generate

from muss.resources.paths import MODELS_DIR
from muss.utils.resources import download_and_extract

from utils.helpers import serialize_to_jsonl, iter_lines
logger = logging.getLogger(__name__)

ALLOWED_MODEL_NAMES = ['muss_en_wikilarge_mined', 'muss_en_mined']


TOKENS_RATIO_DEFAULT = {
    "LengthRatioPreprocessor": 0.9,
    "ReplaceOnlyLevenshteinPreprocessor": 0.8,
    "WordRankRatioPreprocessor": 0.8,
    "DependencyTreeDepthRatioPreprocessor": 0.4,
}

def get_model_path(model_name):
    assert model_name in ALLOWED_MODEL_NAMES
    model_path = MODELS_DIR / model_name
    if not model_path.exists():
        url = f'https://dl.fbaipublicfiles.com/muss/{model_name}.tar.gz'
        extracted_path = download_and_extract(url)[0]
        shutil.move(extracted_path, model_path)
    return model_path


def get_preprocessor(TOKENS_RATIO):
    preprocessors_kwargs = {
        'LengthRatioPreprocessor': {'target_ratio': TOKENS_RATIO["LengthRatioPreprocessor"], 'use_short_name': False},
        'ReplaceOnlyLevenshteinPreprocessor': {
            'target_ratio': TOKENS_RATIO["ReplaceOnlyLevenshteinPreprocessor"],
            'use_short_name': False,
        },
        'WordRankRatioPreprocessor': {
            'target_ratio': TOKENS_RATIO["WordRankRatioPreprocessor"],
            'language': "en",
            'use_short_name': False,
        },
        'DependencyTreeDepthRatioPreprocessor': {
            'target_ratio': TOKENS_RATIO["DependencyTreeDepthRatioPreprocessor"],
            'language': "en",
            'use_short_name': False,
        },
    }
    preprocessors_kwargs['GPT2BPEPreprocessor'] = {}
    return ComposedPreprocessor(get_preprocessors(preprocessors_kwargs))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simplify a file line by line.')
    parser.add_argument('--input-file', type=str, help='', required=True)
    parser.add_argument('--output-file', type=str, help='File to store output')
    parser.add_argument('--model-name', type=str, help='Allowed MUSS models', default="muss_en_wikilarge_mined")
    args = parser.parse_args()

    # 'muss_en_mined'
    exp_dir = get_model_path(args.model_name)
    composed_preprocessor = get_preprocessor(TOKENS_RATIO_DEFAULT)
    
    source_sentences, reference_sentences = list(zip(*[(sent["complex"], sent["simple"]) for sent in iter_lines(args.input_file)]))
    complex_filepath = get_temp_filepath()
    prompt_file = get_temp_filepath()
    write_lines(source_sentences, complex_filepath)
    preprocessed_complex_filepath = get_temp_filepath()
    composed_preprocessor.encode_file(complex_filepath, preprocessed_complex_filepath)
    composed_preprocessor.decode_file(preprocessed_complex_filepath, prompt_file, encoder_filepath=complex_filepath)

    pred_filepath = get_temp_filepath()
    preprocessed_pred_filepath = get_temp_filepath()
    kwargs = {}
    fairseq_generate(preprocessed_complex_filepath, pred_filepath, exp_dir, **kwargs)
    composed_preprocessor.decode_file(pred_filepath, preprocessed_pred_filepath, encoder_filepath=complex_filepath)

    c = 0
    with open(args.output_file, "w", encoding="utf8") if args.output_file != "stdout" else sys.stdout as outf:
        for line in serialize_to_jsonl(source_sentences, [[x] for x in read_lines(preprocessed_pred_filepath)], read_lines(prompt_file), reference_sentences):
            outf.write(f"{line}\n")
            c += 1 

    logger.info(f"Finished inference on {args.input_file}.")
    logger.info(f"Wrote {c} outputs to {args.output_file}")