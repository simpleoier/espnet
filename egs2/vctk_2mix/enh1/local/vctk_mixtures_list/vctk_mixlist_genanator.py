#!/usr/bin/env python

# Copyright 2021  CASIA  (Authors: Jing Shi)
# Apache 2.0
import argparse
import random
from tqdm import tqdm
from pathlib import Path

spk_to_gender={
    "p225": "F", 
    "p226": "M", 
    "p227": "M", 
    "p228": "F", 
    "p229": "F", 
    "p230": "F", 
    "p231": "F", 
    "p232": "M", 
    "p233": "F", 
    "p234": "F", 
    "p236": "F", 
    "p237": "M", 
    "p238": "F", 
    "p239": "F", 
    "p240": "F", 
    "p241": "M", 
    "p243": "M", 
    "p244": "F", 
    "p245": "M", 
    "p246": "M", 
    "p247": "M", 
    "p248": "F", 
    "p249": "F", 
    "p250": "F", 
    "p251": "M", 
    "p252": "M", 
    "p253": "F", 
    "p254": "M", 
    "p255": "M", 
    "p256": "M", 
    "p257": "F", 
    "p258": "M", 
    "p259": "M", 
    "p260": "M", 
    "p261": "F", 
    "p262": "F", 
    "p263": "M", 
    "p264": "F", 
    "p265": "F", 
    "p266": "F", 
    "p267": "F", 
    "p268": "F", 
    "p269": "F", 
    "p270": "M", 
    "p271": "M", 
    "p272": "M", 
    "p273": "M", 
    "p274": "M", 
    "p275": "M", 
    "p276": "F", 
    "p277": "F", 
    "p278": "M", 
    "p279": "M", 
    "p280": "F", 
    "p281": "M", 
    "p282": "F", 
    "p283": "F", 
    "p284": "M", 
    "p285": "M", 
    "p286": "M", 
    "p287": "M", 
    "p288": "F", 
    "p292": "M", 
    "p293": "F", 
    "p294": "F", 
    "p295": "F", 
    "p297": "F", 
    "p298": "M", 
    "p299": "F", 
    "p300": "F", 
    "p301": "F", 
    "p302": "M", 
    "p303": "F", 
    "p304": "M", 
    "p305": "F", 
    "p306": "F", 
    "p307": "F", 
    "p308": "F", 
    "p310": "F", 
    "p311": "M", 
    "p312": "F", 
    "p313": "F", 
    "p314": "F", 
    "p315": "M", 
    "p316": "M", 
    "p317": "F", 
    "p318": "F", 
    "p323": "F", 
    "p326": "M", 
    "p329": "F", 
    "p330": "F", 
    "p333": "F", 
    "p334": "M", 
    "p335": "F", 
    "p336": "F", 
    "p339": "F", 
    "p340": "F", 
    "p341": "F", 
    "p343": "F", 
    "p345": "M", 
    "p347": "M", 
    "p351": "F", 
    "p360": "M", 
    "p361": "F", 
    "p362": "F", 
    "p363": "M", 
    "p364": "M", 
    "p374": "M", 
    "p376": "M", 
    "s5":"F", 
}

def random_sample(idx,spk_list,audios,tag, num_spk,snr_range=5.0, cv_utts_number=15):
    """ Generate one line of the mixtures' list.

    Args:
        idx (int): index in all the mixtures
        spk_list (list): speakers list
        audios (dict): dict with speakers
        tag (str): tr/cv/tt 
        num_spk (int): num of speakers in one mixture.
        snr_range (float, optional): [description]. Defaults to 5.0.
    """

    spks = random.sample(spk_list, num_spk)
    line = ""
    snr_pos = round(random.random() * snr_range/2.0, 5) 
    snr_neg = -1 * snr_pos
    for jdx, spk in enumerate(spks):
        if tag == "tr":
            aim_audios = audios[spk][:-cv_utts_number]
        elif tag == "cv":
            aim_audios = audios[spk][-cv_utts_number:]
        elif tag == "tt":
            aim_audios = audios[spk]
        else: 
            raise KeyError

        line += random.sample(aim_audios,1)[0]
        line += " "
        if jdx == 0:
            line += str(snr_pos)
        elif jdx ==1:
            line += str(snr_neg)
        elif jdx ==2:
            line += str(0)
        line += " "
    return line


def prepare_data(args):
    audiodir = Path(args.vctk_root).expanduser()
    outfile = Path(args.outfile).expanduser().resolve()
    spk_list = [spk_name.name for spk_name in Path(audiodir/"wav48").glob("*")]
    assert len(spk_list)==109, "VCTK should get 109 speakers in total, but got {} now".format(len(spk_list))
    spk_list.remove("p315") # VCTK lost the transcription for Speaker P315, just drop it from now on.

    # Shuffle many times to the spk_list
    random.shuffle(spk_list)
    random.shuffle(spk_list)
    random.shuffle(spk_list)
    spk_list_tr = spk_list[:-args.num_spks_test]
    spk_list_tt = spk_list[-args.num_spks_test:]
    spk_list_tr_gen = [spk_to_gender[spk] for spk in spk_list_tr]
    spk_list_tt_gen = [spk_to_gender[spk] for spk in spk_list_tt]
    print("spk_list_tr_gender: M:{}, F:{}".format(spk_list_tr_gen.count('M'),spk_list_tr_gen.count('F')))
    print("spk_list_tt_gender: M:{}, F:{}".format(spk_list_tt_gen.count('M'),spk_list_tt_gen.count('F')))
    input("Input to continue......")

    audios = {
        spk: [str(sample.relative_to(audiodir)) for sample in audiodir.rglob(spk + "/*." + args.audio_format)]
        for spk in spk_list
    }

    for spk in spk_list:
        random.shuffle(audios[spk])


    with Path(outfile/"spk_list_tr").open("w") as out:
        out.write("\n".join(spk_list_tr))
        out.write("\n")
    with Path(outfile/"spk_list_tt").open("w") as out:
        out.write("\n".join(spk_list_tt))
        out.write("\n")

    for num_spk in args.num_spks:
        for mode in ["tr","cv","tt"]:
            aimfile= "vctk_mix_{}_spk_{}.txt".format(num_spk,mode)
            with Path(outfile/aimfile).open("w") as out:
                for idx_mixture in tqdm(range(getattr(args,"num_mixtures_{}".format(mode)))):
                    # out.write("idx_mixture_{}".format(idx_mixture) + "\n")
                    if mode == "tt": # for open condition
                        aim_list = spk_list_tt
                    else: # for close condition
                        aim_list = spk_list_tr
                    out.write(random_sample(idx_mixture,aim_list,audios,mode,num_spk)+"\n")

                # whether to use trainset augment
                if mode == "tr" and args.trainset_augment_samples>0:
                    aimfile= "vctk_mix_{}_spk_{}_aug{}.txt".format(num_spk,mode,args.trainset_augment_samples)
                    with Path(outfile/aimfile).open("w") as out:
                        for idx_mixture in tqdm(range(getattr(args,"trainset_augment_samples"))):
                            out.write(random_sample(idx_mixture,aim_list,audios,mode,num_spk)+"\n")


    print("Generation of mixture list Finished.")

def get_parser():
    """Argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vctk_root", type=str, help="Path to the VCTK root (v0.80)"
    )
    parser.add_argument("--outfile", type=str,default="./")
    parser.add_argument(
        "--num_spks",
        type=int,
        nargs="+",
        required=True,
        help="Number of speakers in one mixture",
    )
    parser.add_argument(
        "--num_spks_test",
        type=int, 
        default=18, 
        help="number of unknwon speakers from total(109)",
    )

    parser.add_argument("--audio-format", type=str, default="wav")
    parser.add_argument("--num_mixtures_tr", type=int, default=20000)
    parser.add_argument("--num_mixtures_cv", type=int, default=5000)
    parser.add_argument("--num_mixtures_tt", type=int, default=3000)

    parser.add_argument("--trainset_augment_samples", type=int, default=30000) # additional train mixtures for augment
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    prepare_data(args)
