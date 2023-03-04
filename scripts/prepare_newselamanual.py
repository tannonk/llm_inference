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
        # extracted_levs = [get_level_from_full_id(s_id) for s_id in sub_df.sid]
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
    # doc = nlp(sent)
    # tok_list = [str(tok) for tok in doc]
    # tok_sent = ' '.join(tok_list)
    tok_sent = punc_prep(sent)
    return tok_sent


def parse_newsela_data(infile, verbose=True, complex_level=0, simple_level=4):
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
    # root_levs = [get_level_from_full_id([id]) for id in df['cid'].unique()]

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
        if full_prep(src) != full_prep(tgt):
            objects.append({'complex': src, 'simple': tgt, 'complex_level': complex_level, 'simple_level': simple_level})
        else:
            tgt_is_copy_count += 1
    if verbose:
        print(f'Target was a copy {tgt_is_copy_count} times')
    print(f'Finished processing {len(objects)} alignments from level {complex_level} to {simple_level}')
    return objects


if __name__ == '__main__':
    exclude = set(punctuation)
    specials = '~`—$%^#@&*_+=-–<>'
    for c in specials:
        exclude.add(c)
    crowd_testfile = 'data/newsela-auto/newsela-manual/crowdsourced/test.tsv'
    all_testfile = 'data/newsela-auto/newsela-manual/all/test.tsv'
    files = {'crowdsourced_test': crowd_testfile, 'all_test': all_testfile}

    for k,file in files.items():
        outname = f'news_manual_{k}.jsonl'
        all_pairs = []
        for complex_lev in range(4):
            for simp_lev in range(complex_lev+1, 5):
                align_objs = parse_newsela_data(infile=file, simple_level=simp_lev, complex_level=complex_lev)
                all_pairs.extend(align_objs)
        with jsonlines.open(outname, 'w') as jsonl_writer:
            for obj in all_pairs:
                jsonl_writer.write(obj)
