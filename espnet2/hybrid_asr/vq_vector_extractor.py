#!/usr/bin/env python3
import argparse
import logging
import os
from pathlib import Path
import sys
import time
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

from sklearn import datasets
from openTSNE import TSNE

import numpy as np
import parallel_wavegan.models
import soundfile as sf
import torch
from tqdm import trange
from typeguard import check_argument_types
import yaml
from espnet.utils.cli_utils import get_commandline_args
from espnet2.hybrid_asr import tsne_utils as utils
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool
from espnet2.utils.types import str2triple_str
from espnet2.utils.types import str_or_none

vq_active_tokens=[
    78, 480, 68, 86, 70, 345, 178, 375, 102, 123, 355, 279, 188, 401, 267, 432, 410, 209, 97, 166, 23, 132, 215, 508, 163, 89, 305,
    232, 361, 110, 195, 224, 90, 449, 238, 66, 338, 32, 258, 367, 289, 250, 376, 253, 421, 148, 346, 126, 454, 442, 131, 249, 262, 295, 407,
    324, 431, 245, 28, 351, 244, 197, 427, 303, 96, 458, 461, 386, 318, 34, 242, 365, 247, 420, 14, 213, 153, 273, 53, 511, 330, 413, 423,
    39, 382, 171, 502, 424, 288, 466, 419, 451, 310, 331, 221, 340, 263, 274, 326, 313, 463, 417, 92, 105, 227, 434, 452, 481, 499, 99, 484,
    257, 173, 93, 341, 114, 236, 42, 231, 315, 7, 52,
]

EPS = torch.finfo(torch.get_default_dtype()).eps

def vq_decode(pre_trained_model_root, use_gold_spk=False,**kwargs):
    path = Path(pre_trained_model_root) 
    checkpoint=list(path.rglob("*.pkl"))
    assert len(checkpoint) == 1, checkpoint 
    checkpoint=str(checkpoint[0])
    config = list(path.rglob("config.yml"))
    assert len(config) == 1, config
    config=str(config[0])

    # load config
    if config is None:
        dirname = os.path.dirname(checkpoint)
        config = os.path.join(dirname, "config.yml")
    with open(config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    # config.update(vars(vq_args))
    
    # setup model
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    device = torch.device("cpu")
    model_class = getattr(
        parallel_wavegan.models,
        config.get("generator_type", "ParallelWaveGANGenerator"))
    model = model_class(**config["generator_params"])
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu")["model"]["generator"])
    logging.info(f"Loaded model parameters from {checkpoint}.")
    model.remove_weight_norm()
    model = model.eval().to(device)
    print(f"model:{model}")
    print(f"vq:{model.codebook.embedding.weight}")

    vq_vectors = model.codebook.embedding.weight.data.numpy()
    if False:
        np.save("wsj0_vq_numpy.npy", vq_vectors)

    print(f"vq_vec shape:{vq_vectors.shape}")


    embedding = TSNE().fit(vq_vectors)
    # s is the size of scalar point
    utils.plot(embedding, [i for i in range(512)], draw_centers=True, draw_cluster_labels=True, s=15, save_name="plt_selected.jpg") #, colors=utils.MACOSKO_COLORS)

    emb = []
    label = []
    for i in range(512):
        if i in vq_active_tokens:
            emb.append(embedding[i])
            label.append(i)
    emb =np.array(emb)
    label = np.array(label)

    utils.plot(emb, label, draw_centers=True, draw_cluster_labels=True, s=15, save_name="plt.jpg") #, colors=utils.MACOSKO_COLORS)


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="Frontend inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Note(kamo): Use '_' instead of '-' as separator.
    # '-' is confusing if written in yaml.
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument("--pre_trained_model_root", type=str, default="/data3/Espnet_xuankai/espnet/egs2/wsj0_2mix/wsj0_8k_dsf64")
    '''
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers used for DataLoader",
    )

    group = parser.add_argument_group("Input data related")
    group.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        required=True,
        action="append",
    )
    group.add_argument("--key_file", type=str_or_none)
    group.add_argument("--tokens_list", type=str_or_none)
    group.add_argument("--allow_variable_data_keys", type=str2bool, default=False)

    group = parser.add_argument_group("Output data related")
    group.add_argument(
        "--normalize_output_wav",
        type=str2bool,
        default=False,
        help="Whether to normalize the predicted wav to [-1~1]",
    )

    group = parser.add_argument_group("The model configuration related")
    group.add_argument("--hybrid_train_config", type=str, required=True)
    group.add_argument("--hybrid_model_file", type=str, required=True)

    group = parser.add_argument_group("Data loading related")
    group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )
    group = parser.add_argument_group("SeparateSpeech related")
    group.add_argument(
        "--segment_size",
        type=float,
        default=None,
        help="Segment length in seconds for segment-wise speech enhancement/separation",
    )
    group.add_argument(
        "--hop_size",
        type=float,
        default=None,
        help="Hop length in seconds for segment-wise speech enhancement/separation",
    )
    group.add_argument(
        "--normalize_segment_scale",
        type=str2bool,
        default=False,
        help="Whether to normalize the energy of the separated streams in each segment",
    )
    group.add_argument(
        "--show_progressbar",
        type=str2bool,
        default=False,
        help="Whether to show a progress bar when performing segment-wise speech "
        "enhancement/separation",
    )
    group.add_argument(
        "--ref_channel",
        type=int,
        default=None,
        help="If not None, this will overwrite the ref_channel defined in the "
        "separator module (for multi-channel speech processing)",
    )

    group.add_argument(
        "--lm_weight",
        type=float,
        default=0.0,
        help="language model weight.",
    )
    group.add_argument(
        "--beam_size",
        type=int,
        default=10,
        help="language model weight.",
    )
    group.add_argument(
        "--use_beam_search",
        type=str2bool,
        default=False,
        help="Whether to use beam search",
    )
    group.add_argument(
        "--kenlm_file",
        type=str,
        default=None,
        help="kenlm ngram language model path.",
    )
    group.add_argument(
        "--ngram",
        type=int,
        default=0,
        help="n of ngram language model.",
    )
    group.add_argument(
        "--use_gold_spk",
        type=str2bool,
        default=False,
        help="n of ngram language model.",
    )
    '''

    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    vq_decode(**kwargs)


if __name__ == "__main__":
    main()
