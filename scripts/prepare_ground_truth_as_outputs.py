#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Expects a jsonl file with the following format:

{"complex": "...", "simple": ["...", "...", "..."]}

Example call:

    python scripts/prepare_ground_truth_as_outputs.py \
        resources/data/asset/dataset/asset.test.jsonl \
        resources/outputs/ground_truths/asset.test.jsonl

    python scripts/prepare_ground_truth_as_outputs.py \
        resources/data/med-easi/med-easi.test.jsonl \
        resources/outputs/ground_truths/med-easi.test.jsonl

"""

import json
import random
from pathlib import Path
import sys

random.seed(42)

infile = sys.argv[1]
outfile = sys.argv[2]

with open(infile, "r", encoding="utf8") as inf:
    with open(outfile, "w", encoding="utf8") as outf:
        for line in inf:
            data = json.loads(line.strip())
            complex_sent = data["complex"]
            simple_sents = data["simple"]
            if len(simple_sents) > 1:
                target_idx = random.choice(range(len(simple_sents)))
                target = simple_sents.pop(target_idx)
            else:
                target = simple_sents[0]
            outf.write(json.dumps({"source": complex_sent, "model_output": target, "references": simple_sents}) + "\n")
