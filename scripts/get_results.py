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

    python -m scripts.create_checklist

"""

import glob
import itertools
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from pytablewriter import MarkdownTableWriter

EXP_TO_RUN = "exp_configs/rtx"  # Configuration templates for running experiments
EXP_READY = "data/llm_simplification_results"  # Results from the configuration templates
REPORTS_OUT = "data/llm_simplification_results/reports"

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
    "Compression ratio": "c_ratio",
    "Sentence splits": "splits",
    "Levenshtein similarity": "lev_sim",
    "Exact copies": "copies",
    "Additions proportion": "adds",
    "Deletions proportion": "deletes",
    "Lexical complexity score": "lex_complexity"}


def get_results():
    # Check if the test set exists and has the same amount of lines
    files = glob.glob(f"{EXP_READY}/*/*eval")
    files = [file for file in files if "dummy" and "t5-v1-1" not in file]
    output = []
    for file in files:
        parsed_results = get_initial_params(file)
        df = pd.read_csv(file, sep=";")
        file_headers = [c for c in df.columns if "file_id" not in c]
        for head in file_headers:
            score = float(df[head].iloc[0])
            score = round(score, 2)
            parsed_results.append(score)

        output.append(parsed_results)

    columns = get_columns(file_headers)
    final_df = pd.DataFrame.from_records(output, columns=columns)
    save_results_full(final_df)
    save_results_summary(final_df)
    return final_df


def get_filtered_files(files):
    return [file for file in files if "dummy" and "t5-v1-1" and "reports" not in file]


def get_initial_params(file):
    model = Path(file).parent.name
    test, example, prompt, few_n, refs, seed = Path(file).name.split("_")
    few_n = few_n.replace("fs", "")
    refs = refs.replace("nr", "")
    seed = re.search(r'\d+', seed).group()

    return [model, test, example, few_n, prompt, refs, seed]


def get_columns(file_headers):
    columns = ["model", "test", "example", "few_n", "prompt", "refs", "seed"]
    formatted_columns = []
    columns.extend(file_headers)
    for c in columns:
        if c in HEADERS_NAMES.keys():
            formatted_columns.append(HEADERS_NAMES[c])
        else:
            formatted_columns.append(c)
    return formatted_columns


def setup_save(tag):
    results_path = [REPORTS_OUT, f"{REPORTS_OUT}/full", f"{REPORTS_OUT}/summary", "data/checklist/"]
    for p in results_path:
        if not Path(p).exists():
            os.makedirs(p)

    return [r for r in results_path if tag in r][0]


def save_results_full(df):
    path = setup_save("full")
    df = df.groupby(["Model", "Test", "Few(n)", "n", "Prompt", "r"]).mean()
    save_with_format(path, df, "full")


def save_results_summary(df):
    path = setup_save("summary")
    df = df[["Model", "Test", "Prompt", "s", "sari", "fkgl", "F1_bert_ref", "F1_bert_src", "lev_sim", "lex_complexity"]]
    df = df.groupby(["Model", "Test", "Prompt"]).mean()
    save_with_format(path, df, "summary")


def save_with_format(path, df, tag):
    df = df.reset_index().round(2)

    for metric in ["sari", "fkgl", "F1_bert_ref"]:

        sorting_asc = "fkgl" in metric
        df = df.sort_values(by=metric, ascending=sorting_asc)
        df.to_csv(f"{path}/{tag}_results_by_{metric}.csv".lower(), index=False)


def create_checklist():
    files = glob.glob(f"{EXP_READY}/*/*eval")  # Get the list of templates that are already run
    files = get_filtered_files(files)
    parsed_results = []
    for file in files:
        params = get_initial_params(file)
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
    templates = glob.glob(f"{EXP_TO_RUN}/*")  # Get list of config files that are planned to be run
    templates = get_filtered_files(templates)
    templates = [Path(file).stem for file in templates]

    prompts = ["p1", "p2"]
    unique_params = []

    for i in range(0, len(parsed_results[0])):
        a = [result[i] for result in parsed_results]
        unique_params.append(a)

    unique_params[0] = np.append(unique_params[0], templates)  # Models list
    unique_params[4] = np.append(unique_params[4], prompts)  # Prompts list

    unique_params = [np.unique(a) for a in unique_params]

    return unique_params


def print_results(checklist):
    path = setup_save("check")
    outfile = f"{path}/checklist.md"
    checklist = [r.tolist() for r in checklist]

    writer = MarkdownTableWriter(
        headers=["Model", "Test", "Train (few-shot)", "# samples", "Prompt", "# Ref", "Seed", "Done?"],
        value_matrix=checklist,
    )
    writer.dump(outfile)

    with open(f"{path}/checklist.csv", "w") as f:
        for c in checklist:
            index = len(c)-1
            c[index] = c[index].replace(":white_check_mark:", "Yes")
            c[index] = c[index].replace(":heavy_multiplication_x:", "No")
            f.write(f"{','.join(c)}\n")



def main():
    create_checklist()
    get_results()


main()
