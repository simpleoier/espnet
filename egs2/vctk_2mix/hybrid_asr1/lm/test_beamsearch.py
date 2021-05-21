import argparse
import os

import numpy as np
import torch

from espnet2.hybrid_asr.beam_search import BeamSearch
from espnet2.lm.ken_lm import kenlmLM
from espnet2.tasks.lm import LMTask


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm_file", type=str, default=None, help="lm file.")
    parser.add_argument("--ngram", type=int, default=None, help="ngram")
    return parser


def levenshtein(a, b):
    """Calculates the Levenshtein distance between a and b."""
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n

    current = range(n + 1)
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]


def load_data():
    ys_hats1 = torch.from_numpy(
        np.load("./spk1_vq_hats_p241_p303_344_0.03602_265_-0.03602.npy")
    )
    ground_truth1 = "308 71 59 59 59 59 59 59 59 59 71 71 71 58 58 230 230 71 71 58 230 230 230 230 230 230 230 71 59 71 58 58 71 59 450 59 71 71 59 71 230 230 58 71 59 71 58 58 58 58 58 58 71 59 59 59 59 71 71 71 58 230 58 71 59 59 450 450 450 71 58 71 59 59 71 58 230 230 58 58 58 230 58 71 58 58 58 58 58 58 71 59 450 59 71 58 71 71 71 58 58 71 58 230 230 58 230 58 71 58 71 59 59 59 450 450 223 450 450 450 59 59 71 59 59 71 58 58 230 170 230 58 71 71 71 58 230 58 59 59 71 59 59 59 59 59 59 59 71 58 58 230 230 58 58 58 230 230 230 230 230 230 230 58 71 58 58 58 71 59 59 59 450 450 450 450 450 450 450 59 71 230 230 58 230 230 58 58 230 170 170 230 230 71 71 59 71 71 71 59 59 59 450 450 450 59 59 59 59 71 58 230 58 58 58 230 230 230 230 230 230 58 71 59 450 450 450 450 450 59 450 59 59 59 59 59 59 59 59 71 58 230 230 58 58 58 230 230 170 230 230 58 71 59 59 59 59 59 59 450 450 450 59 59 71 58 58 58 58 58 71 58 58 58 71 59 71 58 230 230 58 58 58 71 59 59 59 59 450 450 450 450 450 450 59 71 230 170 170 170 170 230 58 58 230 230 58 230 230 58 59 71 71 59 59 71 59 59 450 59 308 470 41 41 58 470 41 41 41 41 41 41 41 41 470 59 308 308 308 449 223 223 450 449 450 449 308 71 308 308 308 449 449 470 41 41 41 170 41 41 41 170 170 41 470 308 449 308 308 449 449 449 449 308 470 470 470 470 41 41 470 470 41 41 470 41 41 170 170 41 41 41 41 58 470 71 59 449 449 223 223 449 449 449 308 58 58 58 230 58 308 93 447 308 71 71 58 58 230 230 230 71 71 71 71 59 449 449 449 449 449 449 449 308 470 71 71 308 161 399 200 470 58 58 58 58 470 470 58 230 41 351 359 361 161 470 59 59 59 59 449 490 447 447 308 58 230 230 230 230 230 230 58 230 230 230 58 59 59 449 447 447 115 93 452 19 38 464 272 223 152 393 231 242 59 242 14 150 56 450 242 14 170 126 223 152 224 230 126 394 231 340 223 503 334 287 334 308 502 308 386 399 442 442 399 399 361 399 359 459 459 359 202 361 239 239 8 386 357 272 292 292 292 292 292 292 292 241 452 452 397 397 397 397 397 397 442 442 397 287 447 447 386 397 202 386 490 490 490 490 386 386 386".split()

    ys_hats2 = torch.from_numpy(
        np.load("./spk2_vq_hats_p241_p303_344_0.03602_265_-0.03602.npy")
    )
    ground_truth2 = "308 71 58 58 58 58 58 58 59 59 450 59 59 59 71 71 71 58 58 58 71 58 58 71 71 58 58 230 230 71 59 59 71 71 59 59 71 71 71 71 230 230 58 59 59 59 59 59 59 71 71 71 58 230 230 230 58 59 59 59 59 59 59 59 59 59 71 58 230 230 230 230 58 58 58 58 71 71 71 59 450 59 71 59 450 59 71 71 59 59 71 58 58 58 230 230 58 58 71 58 58 230 58 71 59 59 59 450 450 450 59 71 58 58 58 230 58 71 230 230 230 58 71 59 59 59 71 71 71 58 58 71 71 58 230 58 71 59 58 58 230 58 71 59 450 450 450 450 450 59 59 59 58 230 230 58 58 230 58 230 230 230 71 450 450 59 71 58 59 450 450 59 71 58 230 230 71 59 59 71 58 230 230 58 59 59 71 58 71 71 58 58 71 71 71 71 230 230 58 59 71 71 59 450 450 450 450 450 450 59 58 230 230 230 230 170 170 170 230 58 58 58 59 450 450 450 450 450 450 450 450 59 59 59 71 71 71 230 230 230 230 230 58 58 230 230 230 71 59 59 59 59 59 59 450 450 59 59 59 71 59 450 59 71 58 58 71 58 230 170 170 230 58 230 230 230 230 230 58 71 59 450 59 59 450 450 450 450 450 450 450 59 58 58 58 41 290 290 170 170 170 230 230 230 58 58 71 59 450 223 450 450 450 450 449 59 59 71 71 71 58 290 442 359 359 290 170 230 58 71 71 71 71 470 41 470 59 59 59 450 450 59 59 450 450 59 71 58 58 58 71 58 230 230 58 71 58 230 230 58 71 71 58 58 58 59 59 450 59 59 71 59 59 450 59 59 71 71 71 71 58 58 58 58 230 230 230 230 230 58 71 71 58 58 71 59 450 450 59 59 59 450 59 71 59 59 71 58 71 59 71 58 58 230 58 58 230 170 230 230 230 58 58 58 71 59 450 450 450 59 59 59 450 450 59 59 71 71 71 58 230 230 230 170 230 230 230 230 230 58 71 59 59 450 450 450 450 450 450 59 59 450 59 58 58 58 230 230 230 58 58 230 230 58 59 59 59 59 71 71 71 450 59 58 71 71 41 41 115 93 447 447 447 447 490 396 8 361 357 155 38 155 361 203 155 397 412 361 503 152 397 360 241 203 497 377 242 497 340 242 386 325 241 203 14 210 242 115 262 242 41 340 29 290 224 286 152 497 286 56 286 57 262 286 103 71 38 449 387 358 387 252 93 155 490 387 242 58 262 402 41 256 470 412 81 71 242 59 497 56 71 252 241 387 389 287 452 115 298 241 287 490 397 66".split()

    for x, y in zip([ys_hats1, ys_hats2], [ground_truth1, ground_truth2]):
        yield x, y


def main(args):
    token_list = []
    with open("../data/token_list/phn/tokens.txt", "r") as f:
        for line in f.readlines():
            token = line.strip()
            token_list.append(token)

    lm_model = kenlmLM(
        token_list=token_list,
        lm_file=args.lm_file,
        ngram=args.ngram,
    )
    sos = None

    # lm, _ = LMTask.build_model_from_file(
    #     config_file="/export/c05/xkc09/asr/vctk-2mix-vq/exp/lm_train_lm_rnn_phn/config.yaml",
    #     model_file="/export/c05/xkc09/asr/vctk-2mix-vq/exp/lm_train_lm_rnn_phn/valid.loss.ave.pth",
    #     device="cpu",
    # )
    # lm_model = lm.lm
    # sos = len(token_list) - 1

    for ys_hats, ground_truth in load_data():
        ground_truth_tokens = [token_list.index(x) for x in ground_truth]
        gt_len = len(ground_truth_tokens)
        # print('ground_truth_tokens:', len(ground_truth_tokens), ground_truth_tokens[100:130])

        greedy_predict = ys_hats[0].max(-1)[1]
        # print('greedy_prediction:', len(greedy_predict), greedy_predict[100:130])
        print(
            "edit distance of greedy:",
            levenshtein(greedy_predict[:gt_len], ground_truth_tokens),
            "/",
            gt_len,
        )

        for lm_weight in range(1, 4, 1):
            beam_search = BeamSearch(
                lm_model,
                weights=dict(asr=1.0, lm=lm_weight / 10.0),
                beam_size=10,
                token_list=token_list,
                sos=sos,
            )

            nbest_vq_hyps = beam_search(ys_hats[0])
            # print('vq_seqs:', len(nbest_vq_hyps[0].yseq), nbest_vq_hyps[0].yseq[100:130])
            print(
                "edit distance of beamsearch:",
                levenshtein(nbest_vq_hyps[0].yseq[:gt_len], ground_truth_tokens),
                "/",
                gt_len,
            )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
