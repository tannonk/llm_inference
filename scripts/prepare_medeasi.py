#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __Author__ = 'Sweta Agrawal'
# __Email__ = 'sweagraw@umd.edu'
# __Date__ = '2023-04-20'

"""

This script generates JSONL files for the MED-EASI dataset given the raw 
files available at https://huggingface.co/datasets/cbasu/Med-EASi.

Example call:

	python -m scripts.prepare_medeasi

"""

from datasets import load_dataset
from pathlib import Path
import json

data_dir = Path("resources/data/med-easi/")
data_dir.mkdir(parents=True, exist_ok=True)

dataset = load_dataset("cbasu/Med-EASi")

for split in ["train","test", "validation"]:
	outfile = data_dir / f'med-easi.{split}.jsonl'
	c = 0
	with open(outfile, "w", encoding="utf8") as outf:
		for line in dataset[split]:
			d = {"complex": line["Expert"], "simple": [line["Simple"]]}
			outf.write(f"{json.dumps(d, ensure_ascii=False)}\n")
			c += 1
	print(f"Wrote {c} items to {outfile}")