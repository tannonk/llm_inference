#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __Author__ = 'Sweta Agrawal'
# __Email__ = 'sweagraw@umd.edu'
# __Date__ = '2023-04-20'

"""

This script generates JSONL files for the PLAIN ENGLISH LEGAL dataset given the raw 
files available at https://github.com/lauramanor/legal_summarization.

Example call:

    python -m scripts.prepare_plainenglishcorpus

"""


import json
from pathlib import Path

data_dir = Path("resources/data/plainenglishlegal/")

# legal_summarization/all_v1.json
dataset = json.loads(open(data_dir / "all_v1.json").read())

outfile = data_dir / "plain-eng.test.jsonl"
c = 0
with open(outfile, "w", encoding="utf8") as outf:
    for doc_id in dataset:
        d = {"complex": dataset[doc_id]["original_text"], "simple": [dataset[doc_id]["reference_summary"]]}
        outf.write(f"{json.dumps(d, ensure_ascii=False)}\n")
        c += 1
print(f"Wrote {c} items to {outfile}")