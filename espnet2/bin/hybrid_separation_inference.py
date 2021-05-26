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

def vq_decode(utt_id, idx_seq, spk_idx, pre_trained_model_root="/data3/VQ_GAN_codebase/egs/vctk/vc1/", use_gold_spk=False):
    """Run decoding process."""
    vq_seq = torch.LongTensor([idx_to_vq[idx] for idx in idx_seq]).to(idx_seq.device)
    assert vq_seq.shape == idx_seq.shape
    checkpoint=pre_trained_model_root+"exp/train_nodev_all_vctk_conditioned_melgan_vae.v3/checkpoint-5000000steps.pkl"
    config=pre_trained_model_root+"exp/train_nodev_all_vctk_conditioned_melgan_vae.v3/config.yml" 
    verbose=1

    if use_gold_spk: 
        dic = spk_to_netidx
        spk_str = list(dic.keys())[list(dic.values()).index(int(spk_idx))] # find the spk
        print("ori spk idx:", spk_str, spk_idx)
        spk_gentle = spk_to_gender[spk_str]
        spk_gold = gold_spk[spk_gentle]
        spk_gold_idx = dic[spk_gold]
        print("ori spk idx:", spk_gold, spk_gold_idx)
        spk_idx = spk_gold_idx

    # set logger
    if verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    elif verbose > 0:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    else:
        logging.basicConfig(
            level=logging.WARN, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
        logging.warning("Skip DEBUG/INFO messages")

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
    # device = torch.device("cpu")
    device = idx_seq.device
    model_class = getattr(
        parallel_wavegan.models,
        config.get("generator_type", "ParallelWaveGANGenerator"))
    model = model_class(**config["generator_params"])
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu")["model"]["generator"])
    logging.info(f"Loaded model parameters from {checkpoint}.")
    model.remove_weight_norm()
    model = model.eval().to(device)

    utt2spk = None
    if utt2spk is not None:
        assert spk2idx is not None
        with open(utt2spk) as f:
            lines = [l.replace("\n", "") for l in f.readlines()]
        utt2spk = {l.split()[0]: str(l.split()[1]) for l in lines}
        with open(spk2idx) as f:
            lines = [l.replace("\n", "") for l in f.readlines()]
        spk2idx = {l.split()[0]: int(l.split()[1]) for l in lines}

    # start generation
    with torch.no_grad():
        # z = torch.LongTensor(vq_seq).view(1, -1).to(device)
        z = vq_seq.long().view(1, -1).to(device)
        g = None
        if utt2spk is not None:
            spk_idx = spk2idx[utt2spk[utt_id]]
            g = torch.tensor(spk_idx).long().view(1).to(device)
        g = torch.tensor(spk_idx).long().view(1).to(device)
        start = time.time()
        y = model.decode(z, None, g).view(-1).cpu().numpy()
        rtf = (time.time() - start) / (len(y) / config["sampling_rate"])

        # save as PCM 16 bit wav file
        # sf.write(os.path.join("/data3/Espnet_xuankai/espnet/egs2/vctk_2mix/hybrid_asr1/tmp_gen/", f"{utt_id}_gen.wav"),
                    # y, config["sampling_rate"], "PCM_16")
                    

        # report average RTF
        logging.info(f"Finished generation of utterances {utt_id} (RTF = {rtf:.03f}).")

    return y 


class SeparateSpeech:
    """SeparateSpeech class

    Examples:
        >>> import soundfile
        >>> separate_speech = SeparateSpeech("enh_config.yml", "enh.pth")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> separate_speech(audio)
        [separated_audio1, separated_audio2, ...]

    """

    def __init__(
        self,
        hybrid_train_config: Union[Path, str],
        hybrid_model_file: Union[Path, str] = None,
        segment_size: Optional[float] = None,
        hop_size: Optional[float] = None,
        normalize_segment_scale: bool = False,
        show_progressbar: bool = False,
        ref_channel: Optional[int] = None,
        normalize_output_wav: bool = False,
        device: str = "cpu",
        dtype: str = "float32",
        beam_size: int = 10,
        asr_weight: float = 1.0,
        lm_weight: float = 1.0,
        use_beam_search: bool = False,
        kenlm_file: str = None,
        ngram: int = None,
        use_gold_spk: bool = False, 
    ):
        assert check_argument_types()

        # 1. Build Enh model
        hybrid_model, enh_train_args = ASRTask.build_model_from_file(
            hybrid_train_config, hybrid_model_file, device
        )
        hybrid_model.to(dtype=getattr(torch, dtype)).eval()

        self.device = device
        self.dtype = dtype
        self.enh_train_args = enh_train_args
        self.hybrid_model = hybrid_model

        token_list = hybrid_model.token_list
        print("token_list:",token_list)
        weights = dict(
            asr=asr_weight,
            lm=lm_weight,
        )

        self.use_beam_search = use_beam_search
        if use_beam_search:
            # lm_model = kenlmLM(
                # token_list=token_list,
                # lm_file=kenlm_file,
                # ngram=ngram,
            # )
            # sos = None

            lm, _ = LMTask.build_model_from_file(
                model_file="/data3/Espnet_xuankai/espnet/egs2/vctk_2mix/hybrid_asr1/exp/lm4sj/valid.loss.ave.pth",
                config_file="/data3/Espnet_xuankai/espnet/egs2/vctk_2mix/hybrid_asr1/exp/lm4sj/config.yaml",
                device="cuda",
            )
            lm_model = lm.lm
            sos = len(token_list)

            self.beam_search = BeamSearch(
                lm_model=lm_model,
                beam_size=beam_size,
                weights=weights,
                token_list=token_list,
                sos=sos, # n_vocab=len(token_list),
            )

        # only used when processing long speech, i.e.
        # segment_size is not None and hop_size is not None
        self.segment_size = segment_size
        self.hop_size = hop_size
        self.normalize_segment_scale = normalize_segment_scale
        self.normalize_output_wav = normalize_output_wav
        self.show_progressbar = show_progressbar

        self.num_spk = hybrid_model.num_spkrs
        task = "enhancement" if self.num_spk == 1 else "separation"

        self.segmenting = segment_size is not None and hop_size is not None
        if self.segmenting:
            logging.info("Perform segment-wise speech %s" % task)
            logging.info(
                "Segment length = {} sec, hop length = {} sec".format(
                    segment_size, hop_size
                )
            )
        else:
            logging.info("Perform direct speech %s on the input" % task)

    @torch.no_grad()
    def __call__(
        self, speech_mix: Union[torch.Tensor, np.ndarray], fs: int = 8000
    ) -> List[torch.Tensor]:
        """Inference

        Args:
            speech_mix: Input speech data (Batch, Nsamples [, Channels])
            fs: sample rate
        Returns:
            [separated_audio1, separated_audio2, ...]

        """
        assert check_argument_types()

        # Input as audio signal
        if isinstance(speech_mix, np.ndarray):
            speech_mix = torch.as_tensor(speech_mix)

        assert speech_mix.dim() > 1, speech_mix.size()
        batch_size = speech_mix.size(0)
        speech_mix = speech_mix.to(getattr(torch, self.dtype))
        # lenghts: (B,)
        lengths = speech_mix.new_full(
            [batch_size], dtype=torch.long, fill_value=speech_mix.size(1)
        )

        # a. To device
        speech_mix = to_device(speech_mix, device=self.device)
        lengths = to_device(lengths, device=self.device)

        if self.segmenting and lengths[0] > self.segment_size * fs:
            # Segment-wise speech enhancement/separation
            overlap_length = int(np.round(fs * (self.segment_size - self.hop_size)))
            num_segments = int(
                np.ceil((speech_mix.size(1) - overlap_length) / (self.hop_size * fs))
            )
            t = T = int(self.segment_size * fs)
            pad_shape = speech_mix[:, :T].shape
            enh_waves = []
            range_ = trange if self.show_progressbar else range
            for i in range_(num_segments):
                st = int(i * self.hop_size * fs)
                en = st + T
                if en >= lengths[0]:
                    # en - st < T (last segment)
                    en = lengths[0]
                    speech_seg = speech_mix.new_zeros(pad_shape)
                    t = en - st
                    speech_seg[:, :t] = speech_mix[:, st:en]
                else:
                    t = T
                    speech_seg = speech_mix[:, st:en]  # B x T [x C]

                lengths_seg = speech_mix.new_full(
                    [batch_size], dtype=torch.long, fill_value=T
                )
                # b. Enhancement/Separation Forward
                feats, f_lens = self.hybrid_model.encoder(speech_seg, lengths_seg)
                feats, _, _ = self.hybrid_model.separator(feats, f_lens)
                processed_wav = [
                    self.hybrid_model.decoder(f, lengths_seg)[0] for f in feats
                ]
                if speech_seg.dim() > 2:
                    # multi-channel speech
                    speech_seg_ = speech_seg[:, self.ref_channel]
                else:
                    speech_seg_ = speech_seg

                if self.normalize_segment_scale:
                    # normalize the energy of each separated stream
                    # to match the input energy
                    processed_wav = [
                        self.normalize_scale(w, speech_seg_) for w in processed_wav
                    ]
                # List[torch.Tensor(num_spk, B, T)]
                enh_waves.append(torch.stack(processed_wav, dim=0))

            # c. Stitch the enhanced segments together
            waves = enh_waves[0]
            for i in range(1, num_segments):
                # permutation between separated streams in last and current segments
                perm = self.cal_permumation(
                    waves[:, :, -overlap_length:],
                    enh_waves[i][:, :, :overlap_length],
                    criterion="si_snr",
                )
                # repermute separated streams in current segment
                for batch in range(batch_size):
                    enh_waves[i][:, batch] = enh_waves[i][perm[batch], batch]

                if i == num_segments - 1:
                    enh_waves[i][:, :, t:] = 0
                    enh_waves_res_i = enh_waves[i][:, :, overlap_length:t]
                else:
                    enh_waves_res_i = enh_waves[i][:, :, overlap_length:]

                # overlap-and-add (average over the overlapped part)
                waves[:, :, -overlap_length:] = (
                    waves[:, :, -overlap_length:] + enh_waves[i][:, :, :overlap_length]
                ) / 2
                # concatenate the residual parts of the later segment
                waves = torch.cat([waves, enh_waves_res_i], dim=2)
            # ensure the stitched length is same as input
            assert waves.size(2) == speech_mix.size(1), (waves.shape, speech_mix.shape)
            waves = torch.unbind(waves, dim=0)
        else:
            # b. Enhancement/Separation Forward207G
            if not self.hybrid_model.predict_spk:
                encoder_out, encoder_out_lens = self.hybrid_model.encode(speech_mix, lengths) # n_spk * (bs, lens, enc_dim)
                ys_hats = [
                    self.hybrid_model.ce_lo(enc_out) for enc_out in encoder_out
                ] # n_spk * (bs, lens, proj)
                vq_seqs = [ys.max(-1)[1] for ys in ys_hats] # n_spk * (bs,lens)
                print("vq_seqs:", vq_seqs )
                return vq_seqs, None
            else:
                encoder_out, encoder_out_lens, encoder_out_spk = self.hybrid_model.encode(speech_mix, lengths) # n_spk * (bs, lens, enc_dim)
                ys_hats = [
                    self.hybrid_model.ce_lo(enc_out) for enc_out in encoder_out
                ] # n_spk * (bs, lens, proj)
                # print("ys_hats:", ys_hats[0].shape)
                if self.use_beam_search:
                    vq_seqs = []  # nbest vq_hyp seqs.
                    for _, ys in enumerate(ys_hats):
                        bm_result=self.beam_search(ys)
                        assert bm_result.size(1)==ys.size(1)
                        # print("beam_results",bm_result, bm_result.shape)
                        vq_seqs.append(bm_result)
                    # print("vq_seqs:", vq_seqs[0])
                else:
                    vq_seqs = [ys.max(-1)[1] for ys in ys_hats] # n_spk * (bs,lens)
                spk_idx_list = [self.hybrid_model.ce_spk(enc_out).argmax(-1) for enc_out in encoder_out_spk] # n_spk*(bs,)
                # print("vq_seqs:", vq_seqs )
                return vq_seqs, spk_idx_list

        assert len(waves) == self.num_spk, len(waves) == self.num_spk
        assert len(waves[0]) == batch_size, (len(waves[0]), batch_size)
        if self.normalize_output_wav:
            waves = [
                (w / abs(w).max(dim=1, keepdim=True)[0] * 0.9).cpu().numpy()
                for w in waves
            ]  # list[(batch, sample)]
        else:
            waves = [w.cpu().numpy() for w in waves]

        return vq_seqs, waves

    @staticmethod
    @torch.no_grad()
    def normalize_scale(enh_wav, ref_ch_wav):
        """Normalize the energy of enh_wav to match that of ref_ch_wav.

        Args:
            enh_wav (torch.Tensor): (B, Nsamples)
            ref_ch_wav (torch.Tensor): (B, Nsamples)
        Returns:
            enh_wav (torch.Tensor): (B, Nsamples)
        """
        ref_energy = torch.sqrt(torch.mean(ref_ch_wav.pow(2), dim=1))
        enh_energy = torch.sqrt(torch.mean(enh_wav.pow(2), dim=1))
        return enh_wav * (ref_energy / enh_energy)[:, None]

    @torch.no_grad()
    def cal_permumation(self, ref_wavs, enh_wavs, criterion="si_snr"):
        """Calculate the permutation between seaprated streams in two adjacent segments.

        Args:
            ref_wavs (List[torch.Tensor]): [(Batch, Nsamples)]
            enh_wavs (List[torch.Tensor]): [(Batch, Nsamples)]
            criterion (str): one of ("si_snr", "mse", "corr)
        Returns:
            perm (torch.Tensor): permutation for enh_wavs (Batch, num_spk)
        """
        loss_func = {
            "si_snr": self.hybrid_model.si_snr_loss,
            "mse": lambda enh, ref: torch.mean((enh - ref).pow(2), dim=1),
            "corr": lambda enh, ref: (
                (enh * ref).sum(dim=1)
                / (enh.pow(2).sum(dim=1) * ref.pow(2).sum(dim=1) + EPS)
            ).clamp(min=EPS, max=1 - EPS),
        }[criterion]

        _, perm = self.hybrid_model._permutation_loss(ref_wavs, enh_wavs, loss_func)
        return perm


def humanfriendly_or_none(value: str):
    if value in ("none", "None", "NONE"):
        return None
    return humanfriendly.parse_size(value)


def inference(
    output_dir: str,
    batch_size: int,
    dtype: str,
    fs: int,
    ngpu: int,
    seed: int,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    hybrid_train_config: str,
    hybrid_model_file: str,
    allow_variable_data_keys: bool,
    segment_size: Optional[float],
    hop_size: Optional[float],
    normalize_segment_scale: bool,
    show_progressbar: bool,
    ref_channel: Optional[int],
    normalize_output_wav: bool,
    beam_size: int = 10,
    asr_weight: float = 1.0,
    lm_weight: float = 1.0,
    use_beam_search: bool = False,
    kenlm_file: str = None,
    ngram: int = None,
    use_gold_spk: bool = False, 
):
    assert check_argument_types()
    if batch_size > 1:
        raise NotImplementedError("batch decoding is not implemented")
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if ngpu >= 1:
        device = "cuda"
    else:
        device = "cpu"

    # 1. Set random-seed
    set_all_random_seed(seed)

    print("gold_spk:",use_gold_spk)
    # 2. Build separate_speech
    separate_speech = SeparateSpeech(
        hybrid_train_config=hybrid_train_config,
        hybrid_model_file=hybrid_model_file,
        segment_size=segment_size,
        hop_size=hop_size,
        normalize_segment_scale=normalize_segment_scale,
        show_progressbar=show_progressbar,
        ref_channel=ref_channel,
        normalize_output_wav=normalize_output_wav,
        device=device,
        dtype=dtype,
        beam_size=beam_size,
        asr_weight=asr_weight,
        lm_weight=lm_weight,
        use_beam_search=use_beam_search,
        kenlm_file=kenlm_file,
        ngram=ngram,
        use_gold_spk=use_gold_spk,
    )

    # 3. Build data-iterator
    loader = ASRTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=ASRTask.build_preprocess_fn(
            separate_speech.enh_train_args, False
        ),
        collate_fn=ASRTask.build_collate_fn(
            separate_speech.enh_train_args, False
        ),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=True,
    )

    # 4. Start for-loop
    writers = []
    for i in range(separate_speech.num_spk):
        writers.append(
            SoundScpWriter(f"{output_dir}/wavs/{i + 1}", f"{output_dir}/spk{i + 1}.scp")
        )
    asr_writer = DatadirWriter(output_dir)

    for keys, batch in loader:
        assert isinstance(batch, dict), type(batch)
        assert all(isinstance(s, str) for s in keys), keys
        _bs = len(next(iter(batch.values())))
        assert len(keys) == _bs, f"{len(keys)} != {_bs}"
        batch = {k: v for k, v in batch.items() if not k.endswith("_lengths")}

        asr_seqs, spk_idx_list= separate_speech(**batch)
        print("spk_idx_list")

        # print('uttid:',keys[0], ys_hats[0].shape)
        # np.save(open('/tmp/spk1_vq_hats_{}.npy'.format(keys[0]),'wb'), ys_hats[0].data.cpu().numpy())
        # np.save(open('/tmp/spk2_vq_hats_{}.npy'.format(keys[0]),'wb'), ys_hats[1].data.cpu().numpy())
        # 1/0


        if spk_idx_list != None: # spk_Idx inferred by the model
            for spk, (text,spk_idx) in enumerate(zip(asr_seqs,spk_idx_list)):
                # text: list of FloatTensor (bs,T)
                assert text.size(0) == 1
                text = text[0] # only work with batchsize of 1
                print("text.shape:",text.shape)
                for b in range(batch_size):
                    # writers[spk][keys[b]] = fs, w[b]
                    asr_writer["text"+str(spk)][keys[b]] = " ".join(map(str, text.data.cpu().numpy()))
                
                    wave = vq_decode(keys[b], text, spk_idx, use_gold_spk=use_gold_spk)
                    writers[spk][keys[b]] = 24000, wave
        else: # no-prediction of spks
            for spk, text in enumerate(asr_seqs):
                # text: list of FloatTensor (bs,T)
                assert text.size(0) == 1
                text = text[0] # only work with batchsize of 1
                print("text.shape:",text.shape)
                for b in range(batch_size):
                    # writers[spk][keys[b]] = fs, w[b]
                    asr_writer["text"+str(spk)][keys[b]] = " ".join(map(str, text.data.cpu().numpy()))
                
                    wave = vq_decode(keys[b], text, 3) # special Target SPK ID
                    writers[spk][keys[b]] = 24000, wave


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
        "--fs", type=humanfriendly_or_none, default=8000, help="Sampling rate"
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

    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    inference(**kwargs)


if __name__ == "__main__":
    main()
