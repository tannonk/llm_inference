#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __Author__ = 'Tannon Kew'
# __Email__ = 'kew@cl.uzh.ch
# __Date__ = '2023-03-03'

"""

We use the GEM version of the corpus which is Truecased and tokenized.

We detokenize to get the original sentences.

*.norm       tokenized sentences from English Wikipedia
*.turk.0~7   8 reference simplifications by different Amazon Mechanical Turkers 

Example call:

    python -m scripts.prepare_turkcorpus

"""

import json
from pathlib import Path
from utils import *

from sacremoses import MosesDetokenizer

data_dir = Path("data/turkcorpus/data/turkcorpus/GEM")
detokize = True

md = MosesDetokenizer(lang="en")

def detok(text):
    return md.detokenize(text.split())

def gather_complex_simple_sentences(split):
    dataset = []
    src_file = f"{split}.8turkers.tok.norm"
    for src_line in iter_lines(data_dir / src_file):
        src_line = detok(src_line) if detokize else src_line
        dataset.append({"complex": src_line, "simple": []})
    
    for simp_version in range(8):
        tgt_file = f"{split}.8turkers.tok.turk.{simp_version}"
        for i, tgt_line in enumerate(iter_lines(data_dir / tgt_file)):
            tgt_line = detok(tgt_line) if detokize else tgt_line
            dataset[i]["simple"].append(tgt_line)
    return dataset

for split in ["test", "tune"]:
    dataset = gather_complex_simple_sentences(split)
    outfile = data_dir / f"turk.{split}.jsonl"
    c = 0
    with open(outfile, "w", encoding="utf8") as outf:
        for item in dataset:
            outf.write(f"{json.dumps(item, ensure_ascii=False)}\n")
            c += 1
    print(f"Wrote {c} items to {outfile}")