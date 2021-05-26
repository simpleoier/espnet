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

import humanfriendly
import kenlm
import numpy as np
import parallel_wavegan.models
import soundfile as sf
import torch
from tqdm import trange
from typeguard import check_argument_types
import yaml
from espnet.utils.cli_utils import get_commandline_args
from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.fileio.sound_scp import SoundScpWriter
from espnet2.hybrid_asr.beam_search import BeamSearch
from espnet2.hybrid_asr.loss_weights import idx_to_vq, spk_to_gender, spk_to_netidx, gold_spk
from espnet2.lm.ken_lm import kenlmLM
# from espnet2.tasks.enh import
from espnet2.tasks.hybrid_asr import ASRTask 
from espnet2.tasks.lm import LMTask
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool
from espnet2.utils.types import str2triple_str
from espnet2.utils.types import str_or_none



EPS = torch.finfo(torch.get_default_dtype()).eps

def vq_decode(utt_id=0, idx_seq=0, spk_idx=0, pre_trained_model_root="/data3/VQ_GAN_codebase/egs/vctk/vc1/", use_gold_spk=False):
    """Run decoding process."""
    checkpoint=pre_trained_model_root+"exp/train_nodev_all_vctk_conditioned_melgan_vae.v3/checkpoint-5000000steps.pkl"
    config=pre_trained_model_root+"exp/train_nodev_all_vctk_conditioned_melgan_vae.v3/config.yml" 

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
    device = torch.device("cuda")
    model_class = getattr(
        parallel_wavegan.models,
        config.get("generator_type", "ParallelWaveGANGenerator"))
    model = model_class(**config["generator_params"])
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu")["model"]["generator"])
    logging.info(f"Loaded model parameters from {checkpoint}.")
    model.remove_weight_norm()
    model = model.eval().to(device)

    input_scp_file = "../data/tt_max_24k/vq_spk2"
    output_scp_file = "../tmp_gen/"
    if input_scp_file[-1]=='1':
        suffix = '1'
    elif input_scp_file[-1]=='2':
        suffix = '2'
    lines = open(input_scp_file).readlines()
    out_file=open(output_scp_file+"spk{}.scp".format(suffix),'w+')
    for line in lines:
        print(line)
        spk, vq_seq = line.strip().split('  ')
        utt_id = spk
        spk1 = spk[:4]
        spk2 = spk[5:9]
        if input_scp_file[-1]=='1':
            spk = spk1
        elif input_scp_file[-1]=='2':
            spk = spk2
        else: 
            raise ValueError
        spk_idx = spk_to_netidx[spk]
        print(spk, spk_idx)
        idx_seq = torch.tensor([int(s) for s in vq_seq.split(' ')]).to(device)
        vq_seq = idx_seq

        assert vq_seq.shape == idx_seq.shape
        # start generation
        
        with torch.no_grad():
            # z = torch.LongTensor(vq_seq).view(1, -1).to(device)
            z = vq_seq.long().view(1, -1).to(device)
            g = torch.tensor(spk_idx).long().view(1).to(device)
            start = time.time()
            y = model.decode(z, None, g).view(-1).cpu().numpy()
            rtf = (time.time() - start) / (len(y) / config["sampling_rate"])

            # save as PCM 16 bit wav file
            sf.write(os.path.join("/data3/Espnet_xuankai/espnet/egs2/vctk_2mix/hybrid_asr1/tmp_gen/spk"+suffix, f"{utt_id}.wav"), \
                            y, config["sampling_rate"], "PCM_16")
            out_file.write(utt_id+"\t"+os.path.join("/data3/Espnet_xuankai/espnet/egs2/vctk_2mix/hybrid_asr1/tmp_gen/spk"+suffix, f"{utt_id}.wav")+"\n")
            logging.info(f"Finished generation of utterances {utt_id} (RTF = {rtf:.03f}).")

    out_file.close()


if __name__ == "__main__":
    vq_decode()
