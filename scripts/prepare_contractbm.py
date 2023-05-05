#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __Author__ = 'Sweta Agrawal'
# __Email__ = 'sweagraw@umd.edu'
# __Date__ = '2023-05-03'

"""

This script generates JSONL files for the Contract-BM dataset given the raw 
files available at https://developer.ibm.com/exchanges/data/all/split-and-rephrase/.

Example call:

	python -m scripts.prepare_contractbm

"""

from pathlib import Path
import json
import pandas as pd

data_dir = Path("resources/data/contractbm/")
data_dir.mkdir(parents=True, exist_ok=True)

dataset = pd.read_csv(str(data_dir / "split-and-rephrase-data/benchmarks/contract-benchmark.tsv"), sep="\t")

outfile = data_dir / "contract-bm.test.jsonl"
c = 0
with open(outfile, "w", encoding="utf8") as outf:
    for group_name, group_df in dataset.groupby(['complex']):
        d = {"complex": group_name, "simple": group_df.simple.to_list()}
        outf.write(f"{json.dumps(d, ensure_ascii=False)}\n")
        c += 1
print(f"Wrote {c} items to {outfile}")