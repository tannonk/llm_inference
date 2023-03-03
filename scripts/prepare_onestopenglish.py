#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

This script generates JSONL files for the ONESTOPENGLISH dataset given the raw 
files available at https://github.com/nishkalavallabhi/OneStopEnglishCorpus.

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

    python -m scripts.prepare_onestopenglish

"""

import json
from pathlib import Path
from utils import *

data_dir = Path("data/onestopenglish/Sentence-Aligned/")

# OneStopEnglishCorpus/Sentence-Aligned/ADV-ELE.txt

for level_sets in ["ADV-ELE", "ADV-INT", "ELE-INT"]:
    src, tgt = level_sets.split("-")
    src_lines, tgt_lines = [], []
    lines = [line for line in iter_lines(str(data_dir / f"{level_sets}.txt")) if not line.startswith("***")]
    for i, line in enumerate(lines):
        if i % 2 == 0: # even lines are source, starting with 0
            src_lines.append(line)
        else: # odd lines are target
            tgt_lines.append(line)

    assert len(src_lines) == len(tgt_lines)

    outfile = data_dir / f"ose.{src}-{tgt}.jsonl"
    c = 0
    with open(outfile, "w", encoding="utf8") as outf:
        for src_line, tgt_line in zip(src_lines, tgt_lines):
            d = {"complex": src_line, "simple": tgt_line}
            outf.write(f"{json.dumps(d, ensure_ascii=False)}\n")
            c += 1
    print(f"Wrote {c} items to {outfile}")