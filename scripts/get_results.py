#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# __Author__ = 'Laura Vasquez-Rodriguez'
# __Email__ = 'lvasquezcr@gmail.com'
# __Date__ = '2023-04-19'

"""

Script for collecting and aggregating all the results. It also creates a checklist of the pending experiments.
Expects a list of results to be processed and templates (config files).
Results repo (https://github.com/tannonk/llm_simplification_results/) should be cloned under llm_inference/data.

Example usage:

    python -m scripts.get_results \
        --exp_configs exp_configs/cluster \
        --outputs resources/outputs \
        --reports resources/outputs/reports \
        --public reports

"""

import glob
import itertools
import re
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from pytablewriter import MarkdownTableWriter

# user defined args
parser = argparse.ArgumentParser()
parser.add_argument("--exp_configs", type=str, default="exp_configs/cluster", dest="EXP_TO_RUN",
                    help="Path to model configuration templates for running experiments")
parser.add_argument("--outputs", type=str, default="resources/outputs", dest="EXP_READY",
                    help="Path to results from the configuration templates")
parser.add_argument("--reports", type=str, default="resources/outputs/reports", dest="PRIVATE_REPORTS_OUT", help="Path to reports")
parser.add_argument("--public", type=str, default="", dest="PUBLIC_REPORTS_OUT", help="Path to checklist")
args = parser.parse_args()

# Headers renaming for sharing
HEADERS_NAMES = {
    "model": "Model",
    "test": "Test",
    "example": "Few(n)",
    "few_n": "n",
    "refs": "r",
    "seed": "s",
    "prompt": "Prompt",
    "fbert_ref": "F1_bert_ref",
    "fbert_src": "F1_bert_src",
    "lens": "LENS",
    "lens_std": "LENS_std",
    "Compression ratio": "c_ratio",
    "Sentence splits": "splits",
    "Levenshtein similarity": "lev_sim",
    "Exact copies": "copies",
    "Additions proportion": "adds",
    "Deletions proportion": "deletes",
    "Lexical complexity score": "lex_complexity"
}

SUMMARY_COLUMNS = ["Model", "Test", "Prompt", "s", "sari", "fkgl",
                   "F1_bert_ref", "F1_bert_src", "LENS",
                   "lev_sim", "lex_complexity"]

SORT_BY_METRICS = ["sari", "fkgl", "F1_bert_ref", "LENS"]


def validate_env():
    if not Path(args.EXP_READY).exists():
        raise Exception(f"Directory for results does not exist: {args.EXP_READY}")
    if not Path(args.EXP_TO_RUN).exists():
        raise Exception(f"Directory for model configurations does not exist: {args.EXP_TO_RUN}")


def get_results():
    # Check if the test set exists and has the same amount of lines
    files = glob.glob(f"{args.EXP_READY}/*/*eval")
    files = [file for file in files if "dummy" and "t5-v1-1" not in file]
    output = []
    for file in files:
        parsed_results = get_initial_params(file)
        df = pd.read_csv(file, sep=";")
        file_headers = [c for c in df.columns if "file_id" not in c]
        for head in file_headers:
            score = float(df[head].iloc[0])
            parsed_results.append(score)

        output.append(parsed_results)
    columns = get_columns(file_headers)
    final_df = pd.DataFrame.from_records(output, columns=columns)
    save_results_full(final_df)
    save_results_summary(final_df)
    save_results_raw(final_df)
    return final_df


def get_filtered_files(files):
    return [file for file in files if "dummy" and "t5-v1-1" and "reports" not in file]


def get_initial_params(file):
    model = Path(file).parent.name
    hps = Path(file).name.split("_")
    if model in ["ground_truth", "muss"]: # special cases
        # resources/outputs/ground_truth/asset.test.eval
        # resources/outputs/muss_en_mined/asset-test_default.eval
        test, example, prompt, ex_selector, few_n, refs, seed = hps[0], None, None, None, None, None, None
        test = test.replace(".eval", "").replace('.', '-') # strip away extension and replace dots with dashes
        if 'muss_en_wikilarge_mined' in file:
            model = 'muss_en_wikilarge_mined'
        elif 'muss_en_mined' in file:
            model = 'muss_en_mined'
        print(test, example, prompt, ex_selector, few_n, refs, seed)
    elif len(hps) == 6:
        test, example, prompt, few_n, refs, seed = hps
        ex_selector = "random"
    elif len(hps) == 7:
        test, example, prompt, ex_selector, few_n, refs, seed = Path(file).name.split("_")
    else:
        raise ValueError(f"Illegal number of parameters found in the file name: ({len(hps)}) {hps}")

    few_n = few_n.replace("fs", "") if few_n else None
    refs = refs.replace("nr", "") if refs else None
    seed = re.search(r'\d+', seed).group() if seed else None

    return [model, test, example, few_n, prompt, refs, seed, ex_selector]


def get_columns(file_headers):
    columns = ["model", "test", "example", "few_n", "prompt", "refs", "seed", "ex_selector"]
    formatted_columns = []
    columns.extend(file_headers)
    for c in columns:
        if c in HEADERS_NAMES.keys():
            formatted_columns.append(HEADERS_NAMES[c])
        else:
            formatted_columns.append(c)
    return formatted_columns


def setup_save(tag):
    results_path = [
        Path(args.PRIVATE_REPORTS_OUT),
        Path(args.PRIVATE_REPORTS_OUT) / f"full",
        Path(args.PRIVATE_REPORTS_OUT) / f"summary",
        Path(args.PRIVATE_REPORTS_OUT) / f"raw",
    ]
    if args.PUBLIC_REPORTS_OUT:
        results_path.append(Path(args.PUBLIC_REPORTS_OUT))

    for p in results_path:
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)

    return [str(r) for r in results_path if tag in str(r)]


def save_results_full(df, tag="full"):
    paths = setup_save(tag)
    df = df.groupby(["Model", "Test", "Few(n)", "n", "Prompt", "r", "ex_selector"]).mean()
    for path in paths:
        save_with_format(path, df, tag)


def save_results_summary(df, tag="summary"):
    paths = setup_save(tag)
    df = df[SUMMARY_COLUMNS]
    df = df.groupby(["Model", "Test", "Prompt"]).mean()
    for path in paths:
        save_with_format(path, df, tag)

def save_results_raw(df, tag="raw"):
    paths = setup_save(tag)
    df = df.round(4) # use more precision for raw results
    # sort the dataframe for easier git diff
    df = df.sort_values(['Model', 'Test', 'Few(n)', 'n', 'Prompt', 'r', 's', 'ex_selector'])
    for path in paths:
        df.to_csv(f"{path}/{tag}_results.csv".lower(), index=False)
        print(f"Wrote file: {path}/{tag}_results.csv")


def save_with_format(path, df, tag):
    df = df.reset_index().round(2)

    for metric in SORT_BY_METRICS:
        sorting_asc = "fkgl" in metric
        df = df.sort_values(by=metric, ascending=sorting_asc)
        df.to_csv(f"{path}/{tag}_results_by_{metric}.csv".lower(), index=False)
        print(f"Wrote file: {path}/{tag}_results_by_{metric}.csv")

def create_checklist():
    files = glob.glob(f"{args.EXP_READY}/*/*eval")  # Get the list of templates that are already run
    files = get_filtered_files(files)
    parsed_results = []
    for file in files:
        params = get_initial_params(file)
        if any(param is None for param in params): # Skip ground truth
            continue
        params = [p for p in params if 'val' not in p] # remove validation set
        parsed_results.append(params)

    uniq_params = get_unique_params(parsed_results)  # Combine templates, experiments and create an unique list
    checklist = [list(i) for i in itertools.product(*uniq_params)]
    for i, item in enumerate(checklist):
        if item in parsed_results:
            checklist[i] = np.append(item, ":white_check_mark:")
        else:
            checklist[i] = np.append(item, ":heavy_multiplication_x:")

    print_results(checklist)

    return checklist


def get_unique_params(parsed_results):
    templates = glob.glob(f"{args.EXP_TO_RUN}/*")  # Get list of config files that are planned to be run
    templates = get_filtered_files(templates)
    templates = [Path(file).stem for file in templates]

    unique_params = []
    for i in range(0, len(parsed_results[0])):
        a = [result[i] for result in parsed_results]
        unique_params.append(a)
    
    unique_params[0] = np.append(unique_params[0], templates)  # Models list

    unique_params = [np.unique(a) for a in unique_params]

    return unique_params


def print_results(checklist):
    paths = [f"{args.PRIVATE_REPORTS_OUT}"] # add the private reports folder to the list
    if args.PUBLIC_REPORTS_OUT:
        paths.append(f"{args.PUBLIC_REPORTS_OUT}")

    checklist = [r.tolist() for r in checklist]

    writer = MarkdownTableWriter(
        headers=["Model", "Test", "# samples", "Prompt", "# Ref", "Seed", "Strategy", "Done?"],
        value_matrix=checklist,
    )
    for path in paths:
        outfile = f"{path}/checklist.md"
        writer.dump(outfile)
        print(f"Wrote file: {outfile}")
    
        with open(f"{path}/checklist.csv", "w") as f:
            for c in checklist:
                index = len(c) - 1
                c[index] = c[index].replace(":white_check_mark:", "Yes")
                c[index] = c[index].replace(":heavy_multiplication_x:", "No")
                f.write(f"{','.join(c)}\n")
        print(f"Wrote file: {path}/checklist.csv")

def main():
    validate_env()
    create_checklist()
    get_results()


main()
