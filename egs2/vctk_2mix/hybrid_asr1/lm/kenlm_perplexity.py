# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np
import kenlm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm_file", default=None, type=str, help="kenLM arpa file.")
    parser.add_argument("--test_text", default=None, type=str, help="test text file.")

    return parser


def main(args):
    assert os.path.exists(args.lm_file)
    model = kenlm.Model(args.lm_file)

    assert os.path.exists(args.test_text)

    total_nll = 0
    total_tokens = 0
    with open(args.test_text, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            lst = line.split()
            text_line = ' '.join(lst[1:])
            # score1 = model.score(text_line, bos = False, eos = False)
            score = model.score(text_line, bos = True, eos = True)
            total_nll += score
            total_tokens += len(lst) - 1 + 2
    
    ppl = 10 ** (-total_nll / total_tokens)
    print("PPL:", ppl)


if __name__=="__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)