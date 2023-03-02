#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

This script generates JSONL files for the ASSET dataset given the raw 
files available at https://github.com/facebookresearch/asset.

The output format is JSONL with each line containing a dictionary-like object 
with the following structure:

{
    "complex": "The bank then lends these deposits to borrowers."
    "simple": [
        "The bank lends the deposits to borrowers.", 
        "The bank lends deposits to borrowers.", 
        "The bank then lends these deposits to people.", 
        "The bank gives these deposits to borrowers.",
        ...
    ]
}

Example call:

    python -m scripts.prepare_asset

"""

import json
from pathlib import Path
from utils import *

asset_dir = Path("data/asset/dataset")

def gather_complex_simple_sentences(split):
    dataset = []
    src_file = f"asset.{split}.orig"
    for src_line in iter_lines(asset_dir / src_file):
        dataset.append({"complex": src_line, "simple": []})
    
    for simp_version in range(10):
        tgt_file = f"asset.{split}.simp.{simp_version}"
        for i, tgt_line in enumerate(iter_lines(asset_dir / tgt_file)):
            dataset[i]["simple"].append(tgt_line)
    return dataset

for split in ["test", "valid"]:
    dataset = gather_complex_simple_sentences(split)
    outfile = asset_dir / f"asset.{split}.jsonl"
    c = 0
    with open(outfile, "w", encoding="utf8") as outf:
        for item in dataset:
            outf.write(f"{json.dumps(item, ensure_ascii=False)}\n")
            c += 1
    print(f"Wrote {c} items to {outfile}")