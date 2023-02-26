#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from utils import *

asset_dir = Path("data/asset/dataset")

def gather_complex_simple_sentences(split):
    dataset = []
    src_file = f"asset.{split}.orig"
    # print(asset_dir / src_file)
    for src_line in iter_lines(asset_dir / src_file):
        dataset.append({"complex": src_line})
    
    for simp_version in range(10):
        tgt_file = f"asset.{split}.simp.{simp_version}"
        # print(asset_dir / tgt_file)
        for i, tgt_line in enumerate(iter_lines(asset_dir / tgt_file)):
            dataset[i][f"simple_{simp_version}"] = tgt_line

    return dataset

for split in ["test", "valid"]:
    dataset = gather_complex_simple_sentences(split)
    outfile = asset_dir / f"{split}.jsonl"
    c = 0
    with open(outfile, "w", encoding="utf8") as outf:
        for item in dataset:
            outf.write(json.dumps(item))
            c += 1
    print(f"Wrote {c} items to {outfile}")