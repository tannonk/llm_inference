#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Example call:

    python -m scripts.estimate_cohere_costs resources/outputs/cohere-command-light

    python -m scripts.estimate_cohere_costs resources/outputs/openai-gpt-3.5-turbo

"""

import sys
import json
import math
from pathlib import Path

model_dir = sys.argv[1]

for infile in Path(model_dir).glob('*.jsonl'):
    char_count = 0
    print(f'Processing {infile}')
    with open(infile, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                char_count += len(data['input_prompt'])
                char_count += len(data['model_output'])
    print(f'Total number of characters: {char_count}')
    print(f'Estimated generation units: {math.ceil(char_count/1000)}')
    print()