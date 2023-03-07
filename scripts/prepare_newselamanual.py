#!/usr/bin/env python
# coding: utf-8


"""
Code slightly modified from Tannon Kew's
https://github.com/ZurichNLP/SimpleFUDGE/blob/master/ats_data/extract_aligned_sents_wiki_newsela_manual.py

Modifier: Alison Chi
"""

import os
from typing import List
from string import punctuation
import re
from nltk import word_tokenize
import csv
import jsonlines
import pandas as pd


def get_level_from_full_id(id: List):
    """
    extracts simplification level from a sentence identifier used in
    newsela-manual/wiki-manual, e.g. `['chinook-recognition.en-1-0-0']`
    Note: we handle only lists of ids to simplify cases of
    m:n and 1:1 alignments
    """
    if not isinstance(id, list):
        raise RuntimeError(f'Expected as list of id(s), but got {type(id)}')
    title, id = id[0].split('.')
    return int(id.split('-')[1])


def dedup_sents(lst):
    """
    Removes duplicate sentences from a set of aligned sentences keeping order
    """
    no_dupes = []
    [no_dupes.append(elem) for elem in lst if not no_dupes.count(elem)]
    return no_dupes


def get_title_from_full_id(id: List):
    """
    extracts title from a sentence identifier used in
    newsela-manual/wiki-manual, e.g. `['chinook-recognition.en-1-0-0']`
    Note: we handle only lists of ids to simplify cases of
    m:n and 1:1 alignments
    """
    if not isinstance(id, list):
        raise RuntimeError(f'Expected as list of id(s), but got {type(id)}')
    title, id = id[0].split('.')
    return title


def extract_pairs(df, cid: List, tgt_level: int):
    """
    Recursive function to extract aligned sentences from the
    next adjacent simplification level
    """
    if not cid:
        return None

    c_level = get_level_from_full_id(cid)
    if c_level == tgt_level:  # found aligned simple units
        sub_df = df[df['sid'].isin(cid)]
        sents = ' '.join(dedup_sents(sub_df.ssent.tolist()))
        return sents

    else:  # recursion
        sub_df = df[df['cid'].isin(cid)]
        next_cid = sub_df.sid.tolist()
        return extract_pairs(df, next_cid, tgt_level)


def punc_prep(sent: str):
    """
    Remove all punctuation from a given sentence
    """
    new_sent = ''
    for ch in sent:
        if ch not in exclude:
            new_sent += ch
        else:
            new_sent += ' '
    new_sent = re.sub(r"['-]+\ *", " ", new_sent)
    new_sent = re.sub(' +', ' ', new_sent).strip()
    return new_sent


def full_prep(sent: str):
    """
    Tokenize a sentence and remove punctuation. Not using spacy here since it's so slow,
    but it will be more accurate that way.
    """
    sent = ' '.join(word_tokenize(sent)).lower()
    tok_sent = punc_prep(sent)
    return tok_sent


def parse_newsela_data(infile, verbose=True, complex_level=0, simple_level=4, filter_copies=False):
    """
    Processes annotated alignment file from Newsela-Manual (e.g. `newsela-auto/newsela-manual/all/test.tsv`)
    """

    df = pd.read_csv(infile, sep='\t', header=None, names=['label', 'sid', 'cid', 'ssent', 'csent'], quoting=csv.QUOTE_NONE)
    if verbose:
        print(f'DF contains {len(df)} items')

    df = df[df['label'] != 'notAligned']  # filter all not aligned sentences
    if verbose:
        print(f'Removed `notAligned`. DF contains {len(df)} items')

    root_nodes = [[id] for id in df['cid'].unique() if get_level_from_full_id([id]) == complex_level]

    if verbose:
        print(len(root_nodes))
        print(root_nodes[:5], '...')

    # collect alignments
    alignments = []
    for root_node in root_nodes:
        sub_df = df[(df['cid'].isin(root_node))]

        csents = dedup_sents(sub_df.csent.tolist())
        if len(set(csents)) != len(csents):
            raise RuntimeError
        try:
            src = ' '.join(csents)
        except TypeError:
            print(f'Could not parse sentence {root_node} as string. Check for unclosed quotations!', csents)
            continue

        tgt = extract_pairs(df, root_node, simple_level)
        if tgt:
            alignments.append((src, tgt, get_title_from_full_id(root_node)))

    objects = []
    tgt_is_copy_count = 0
    for src, tgt, _ in alignments:
        if not filter_copies or full_prep(src) != full_prep(tgt):
            objects.append({'complex': src, 'simple': tgt, 'complex_level': complex_level, 'simple_level': simple_level})
        elif filter_copies:
            tgt_is_copy_count += 1
    if verbose and filter_copies:
        print(f'Target was a copy {tgt_is_copy_count} times')
    print(f'Finished processing {len(objects)} alignments from level {complex_level} to {simple_level}')
    return objects


def get_inp_ref_format(all_objs, num_refs=4):
    inp_set = set()
    formatted_objs = []
    ob_dict = {}
    for ob in all_objs:
        ori_tup = (ob['complex'], ob['complex_level'])
        if ori_tup not in inp_set:
            inp_set.add(ori_tup)
        ob_dict[ob['complex']] = {'complex_level': ob['complex_level'], 'simple_level': [], 'simple': []}
    for ob in all_objs:
        if (ob['complex'], ob['complex_level']) in inp_set:
            ob_dict[ob['complex']]['simple_level'].append(ob['simple_level'])
            ob_dict[ob['complex']]['simple'].append(ob['simple'])
    for key, val in ob_dict.items():
        # Sadly, EASSE only allows references that are consistent in length.
        if not num_refs or len(val['simple']) == num_refs:
            formatted_objs.append({'complex': key, 'simple': val['simple'],
                                   'complex_level': val['complex_level'], 'simple_level': val['simple_level']})
    return formatted_objs


def verify_x_in_y(data_x, data_y):
    y_set = set()
    y_df = pd.read_csv(data_y, sep='\t', header=None, names=['label', 'sid', 'cid', 'ssent', 'csent'],
                       quoting=csv.QUOTE_NONE)
    y_df = y_df[y_df['label'] != 'notAligned']  # filter all not aligned sentences
    x_df = pd.read_csv(data_x, sep='\t', header=None, names=['label', 'sid', 'cid', 'ssent', 'csent'],
                       quoting=csv.QUOTE_NONE)
    x_df = x_df[x_df['label'] != 'notAligned']
    for i in range(len(y_df['ssent'])):
        y_set.add((y_df['sid'].tolist()[i], y_df['cid'].tolist()[i],
                   y_df['ssent'].tolist()[i], y_df['csent'].tolist()[i], y_df['label'].tolist()[i]))
    num_x_not_in_y = 0
    for i in range(len(x_df['ssent'])):
        x_tup = (x_df['sid'].tolist()[i], x_df['cid'].tolist()[i],
                 x_df['ssent'].tolist()[i], x_df['csent'].tolist()[i], x_df['label'].tolist()[i])
        if x_tup not in y_set:
            num_x_not_in_y += 1
    return num_x_not_in_y



if __name__ == '__main__':
    # For more efficient preprocessing
    exclude = set(punctuation)
    specials = '~`—$%^#@&*_+=-–<>'
    for c in specials:
        exclude.add(c)

    all_testfile = 'data/newsela-auto/newsela-manual/all/test.tsv'
    all_valfile = 'data/newsela-auto/newsela-manual/all/dev.tsv'
    files = {'all_test': all_testfile, 'all_val': all_valfile}
    out_dir = 'data/newsela-auto'
    for k, file in files.items():
        outname = f'news_manual_{k}.jsonl'
        all_pairs = []
        for complex_lev in range(1):  # only looking at source level 0
            for simp_lev in range(complex_lev+1, 5):
                align_objs = parse_newsela_data(infile=file, simple_level=simp_lev, complex_level=complex_lev)
                all_pairs.extend(align_objs)
        formatted_objects = get_inp_ref_format(all_objs=all_pairs, num_refs=4)
        with jsonlines.open(os.path.join(out_dir, outname), 'w') as jsonl_writer:
            for obj in formatted_objects:
                jsonl_writer.write(obj)
