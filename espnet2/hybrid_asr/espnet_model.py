from contextlib import contextmanager
from distutils.version import LooseVersion
import logging
import os
import time
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
import yaml

import numpy as np
import parallel_wavegan.models
import soundfile as sf
import torch
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.e2e_asr_mix import PIT
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.hybrid_asr.loss_weights import idx_to_vq_max as idx_to_vq
from espnet2.hybrid_asr.loss_weights import normedWeights
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


def pick_longer_utt(ref1,ref2,lenghts1,lenghts2):
    """ ref1,ref2: shape of [bs, label_lens]
        len1,len2: shape of [bs]
    """
    longer_ref = []
    for s1,s2,l1,l2 in zip(ref1,ref2,lenghts1,lenghts2):
        assert l1==l2, (s1.shape, s2.shape, l1,l2)
        s1_nopad = s1[:l1]
        s2_nopad = s2[:l2]
        s1_mask =  (s1_nopad != 0).int() 
        s2_mask =  (s2_nopad != 0).int()
        if s1_mask[-50:].sum() > s2_mask[-50:].sum():
            longer_ref.append(s1)
            # print("[s1] s2 last 30:", s1_nopad[-30:], s2_nopad[-30:])
        else:
            longer_ref.append(s2)
            # print("s1 [s2] last 30:", s1_nopad[-30:], s2_nopad[-30:])
    longer_ref = torch.stack(longer_ref,dim=0) #bs,label_lens
    assert longer_ref.shape == ref1.shape ==ref2.shape
    return longer_ref

class ESPnetHybridASRModel(AbsESPnetModel):
    """Hybrid ASR model"""

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        decoder: None,
        rnnt_decoder: None,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = False,
        report_wer: bool = False,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        num_spkrs: int = 2,
        chunk_size: int = -1,
        use_label_weights: bool = True,
        only_longer_ref: bool = False,
        predict_spk: bool = False,
    ):
        assert check_argument_types()
        assert rnnt_decoder is None, "Not implemented"

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.token_list = token_list.copy()
        self.chunk_size = chunk_size

        self.num_spkrs = num_spkrs #encoder.num_spkrs
        self.pit = PIT(self.num_spkrs)

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.encoder = encoder
        # Separation done in the middle of encoder or at the output layer.
        if getattr(encoder, 'num_spkrs', None) is not None:
            self.ce_lo = torch.nn.Linear(encoder.output_size(), vocab_size)
        else:
            self.ce_lo = torch.nn.ModuleList([
                torch.nn.Linear(encoder.output_size(), vocab_size),
                torch.nn.Linear(encoder.output_size(), vocab_size),
            ])
        if use_label_weights:
            self.cross_entropy = torch.nn.CrossEntropyLoss(
                weight=torch.FloatTensor(normedWeights),
                ignore_index=ignore_id,
                reduce=False,
                reduction="none"
            )
        else:
            self.cross_entropy = torch.nn.CrossEntropyLoss(
                ignore_index=ignore_id,
                reduce=False,
                reduction="none"
            )
        

        self.dropout_rate = 0.0
        self.cut_begin_end = False
        self.only_longer_ref = only_longer_ref
        
        self.predict_spk = predict_spk
        if self.predict_spk:
            spk_total_num=108 # all spks in vctk (except the p315)
            self.ce_spk = torch.nn.Linear(encoder.output_size(), spk_total_num)
            self.cross_entropy_spk = torch.nn.CrossEntropyLoss(
                ignore_index=ignore_id,
                reduce=False,
                reduction="none"
            )

        if report_cer or report_wer:
            self.error_calculator = ErrorCalculator(
                token_list, sym_space, sym_blank, report_cer, report_wer
            )
        else:
            self.error_calculator = None

    def forward(
        self,
        speech_mix: torch.Tensor,
        speech_mix_lengths: torch.Tensor,
        phn_ref1: torch.Tensor,
        phn_ref1_lengths: torch.Tensor,
        phn_ref2: torch.Tensor,
        phn_ref2_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            target: (Batch, Length)
            target_lengths: (Batch,)
        """
        if self.predict_spk:
            assert "spk1_str" in kwargs
            spk1_idx = kwargs["spk1_str"]
            spk2_idx = kwargs["spk2_str"]
        else:
            spk1_idx, spk2_idx = None, None

        assert phn_ref1_lengths.dim() == 1, phn_ref1_lengths.shape
        # Check that batch_size is unified
        assert (
            speech_mix.shape[0]
            == speech_mix_lengths.shape[0]
            == phn_ref1.shape[0]
            == phn_ref1_lengths.shape[0]
            == phn_ref2.shape[0]
            == phn_ref2_lengths.shape[0]
        ), (
            speech_mix.shape,
            speech_mix_lengths.shape,
            phn_ref1.shape,
            phn_ref1_lengths.shape,
            phn_ref2.shape,
            phn_ref2_lengths.shape,
        )

        if self.chunk_size < 0:
            return self.forward_seq(
                speech_mix, speech_mix_lengths, phn_ref1, phn_ref1_lengths, phn_ref2, phn_ref2_lengths, spk1_idx, spk2_idx
            )
        else:
            return self.forward_chunk(
                speech_mix, speech_mix_lengths, phn_ref1, phn_ref1_lengths, phn_ref2, phn_ref2_lengths
            )

    def forward_seq(
        self,
        speech_mix: torch.Tensor,
        speech_mix_lengths: torch.Tensor,
        phn_ref1: torch.Tensor,
        phn_ref1_lengths: torch.Tensor,
        phn_ref2: torch.Tensor,
        phn_ref2_lengths: torch.Tensor,
        spk1_idx: None,
        spk2_idx: None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            target: (Batch, Length)
            target_lengths: (Batch,)
            spk1_idx; None or (Batch, )
            spk2_idx; None or (Batch, )
        """
        batch_size = speech_mix.shape[0]

        # for data-parallel
        targets = [
            phn_ref1[:, : phn_ref1_lengths.max()],
            phn_ref2[:, : phn_ref2_lengths.max()],
        ]
        targets_lengths = [
            phn_ref1_lengths,
            phn_ref2_lengths,
        ]

        if self.only_longer_ref:
            longer_phn_ref = pick_longer_utt(phn_ref1,phn_ref2,phn_ref1_lengths,phn_ref2_lengths)
            longer_target = longer_phn_ref
            longer_target_lengths =  phn_ref1_lengths

        # 1. Encoder
        if self.predict_spk:
            encoder_out, encoder_out_lens, encoder_out_spk, encoder_out_lens_spk = self.encode(speech_mix, speech_mix_lengths) # n_spk * (bs, lens, enc_dim)
        else:
            encoder_out, encoder_out_lens= self.encode(speech_mix, speech_mix_lengths) # n_spk * (bs, lens, enc_dim)

        ys_hats, ys_hats_lengths, phn_ref1, phn_ref2 = self._compute_output_layer(
            encoder_out,
            encoder_out_lens,
            phn_ref1,
            phn_ref2,
            phn_ref1_lengths,
            phn_ref2_lengths,
        )
        targets = [phn_ref1, phn_ref2]

        if self.only_longer_ref:
            loss_ce = self._calc_ce_loss(ys_hats[0],ys_hats_lengths[0],longer_target,longer_target_lengths).mean()
            all_ys_hats = ys_hats[0]
            all_targets = longer_target
            assert all_ys_hats.shape[:2] == all_targets.shape, (all_ys_hats.shape, all_targets.shape)
            acc_ce = th_accuracy(
                all_ys_hats.view(-1, self.vocab_size),
                all_targets,
                ignore_label=self.ignore_id,
            )
            print("ys_hats:", ys_hats[0].shape, ys_hats[0][0].max(-1)[1][800:820])
            print("longer_targets:", all_targets[0][800:820].data.cpu().numpy())
            print("predict:", all_ys_hats.max(-1)[1][0][800:820].data.cpu().numpy(), '\n')
            loss = loss_ce

            stats = dict(
                loss=loss.detach(),
                acc=acc_ce,
            )
            pad_pred = all_ys_hats.view(
                all_targets.size(0), all_targets.size(1), all_ys_hats.size(-1)
            ).argmax(2)
            acc_unit = (pad_pred == all_targets)
            print("acc_unit", acc_unit.shape, acc_unit.sum(1)/acc_unit.size(1))

            # force_gatherable: to-device and to-tensor if scalar for DataParallel
            loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
            return loss, stats, weight

        loss_ce_perm = torch.stack(
            [
                self._calc_ce_loss(
                    ys_hats[i // self.num_spkrs],
                    ys_hats_lengths[i // self.num_spkrs],
                    targets[i % self.num_spkrs],
                    targets_lengths[i % self.num_spkrs],
                )
                for i in range(self.num_spkrs ** 2)
            ],
            dim=1
        ) # (bs, n_spk**2)

        loss_ce, min_perm = self.pit.pit_process(loss_ce_perm) # min_perm: bs*[0, 1] or bs*[1, 0]

        all_targets = []
        for i in range(self.num_spkrs):
            for n in range(batch_size):
                all_targets.append(targets[min_perm[n][i]][n])
        all_targets = torch.stack(all_targets, dim=0) # (bs*spk, lens)
        all_ys_hats = torch.cat(ys_hats, dim=0)
        acc_ce = th_accuracy(
            all_ys_hats.view(-1, self.vocab_size),
            all_targets,
            ignore_label=self.ignore_id,
        )
        print("ys_hats:", ys_hats[0].shape, ys_hats[0][0].max(-1)[1][800:820])
        print("targets:", all_targets[0][800:820].data.cpu().numpy())
        print("predict:", all_ys_hats.max(-1)[1][0][800:820].data.cpu().numpy())
        # More detailed acc info:
        pad_pred = all_ys_hats.view(
            all_targets.size(0), all_targets.size(1), all_ys_hats.size(-1)
        ).argmax(2)
        acc_unit = (pad_pred == all_targets)
        print("acc_unit", acc_unit.shape, acc_unit.sum(1)/acc_unit.size(1))
        '''
        if acc_unit.sum(1)[0]/acc_unit.size(1)>0.7:
            vq_decode('{}_0'.format(acc_unit.sum(1)[0]/acc_unit.size(1)), all_ys_hats[0].max(-1)[1])
            vq_decode('{}_1'.format(acc_unit.sum(1)[1]/acc_unit.size(1)), all_ys_hats[1].max(-1)[1])
            vq_decode('{}_0_ref'.format(acc_unit.sum(1)[0]/acc_unit.size(1)), all_targets[0])
            vq_decode('{}_1_ref'.format(acc_unit.sum(1)[1]/acc_unit.size(1)), all_targets[1])
        elif acc_unit.sum(1)[0]/acc_unit.size(1)>0.7:
            vq_decode('{}_0'.format(acc_unit.sum(1)[0]/acc_unit.size(1)), all_ys_hats[0].max(-1)[1])
            vq_decode('{}_1'.format(acc_unit.sum(1)[1]/acc_unit.size(1)), all_ys_hats[1].max(-1)[1])
            vq_decode('{}_0_ref'.format(acc_unit.sum(1)[0]/acc_unit.size(1)), all_targets[0])
            vq_decode('{}_1_ref'.format(acc_unit.sum(1)[1]/acc_unit.size(1)), all_targets[1])
        '''

        if self.predict_spk:
            targets_spk = [spk1_idx, spk2_idx]
            all_targets_spk = []
            for i in range(self.num_spkrs):
                for n in range(batch_size):
                    all_targets_spk.append(targets_spk[min_perm[n][i]][n])
            all_targets_spk = torch.stack(all_targets_spk, dim=0) # (bs*spk, )
            print('encoder_out_spk:', encoder_out_spk[0].shape)
            ys_hats_spk = [
                self.ce_spk(F.dropout(enc_out, p=self.dropout_rate)) for enc_out in encoder_out_spk
            ] # n_spk * (bs, proj)
            all_ys_hats_spk = torch.cat(ys_hats_spk, dim=0) #(bs*spk, proj)
            # print("targets_spk:", all_targets_spk.squeeze().data.cpu().numpy())
            # print("predict_spk:", all_ys_hats_spk.argmax(-1).data.cpu().numpy())
            loss_ce_spk = self.cross_entropy_spk(all_ys_hats_spk, all_targets_spk.squeeze()).mean()

            acc_ce_spk = th_accuracy(
                all_ys_hats_spk.view(-1, 108),
                all_targets_spk,
                ignore_label=self.ignore_id,
            )
            loss_ce = loss_ce + 100 * loss_ce_spk

        predictions = all_ys_hats.view(-1, self.vocab_size).argmax(-1).cpu().numpy()
        (uniq, counts) = np.unique(predictions, return_counts=True)
        token_variety = uniq.size
        loss = loss_ce

        stats = dict(
            loss=loss.detach(),
            acc=acc_ce,
            token_variety = token_variety,
        )

        if self.predict_spk:
            stats.update(dict(acc_spk=acc_ce_spk))
            stats.update(dict(loss_ce_spk=loss_ce_spk.detach()))
        print('\n\n')

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def forward_chunk(
        self,
        speech_mix: torch.Tensor,
        speech_mix_lengths: torch.Tensor,
        phn_ref1: torch.Tensor,
        phn_ref1_lengths: torch.Tensor,
        phn_ref2: torch.Tensor,
        phn_ref2_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            target: (Batch, Length)
            target_lengths: (Batch,)
        """
        batch_size = speech_mix.shape[0]

        loss = []
        corr = []
        all_ys_hats_record = []
        n_chunks = (phn_ref1.shape[1] + self.chunk_size - 1) // self.chunk_size
        end_flag = False
        for i in range(n_chunks):
            if getattr(self.frontend, 'stft', None) is None:
                start = i * self.chunk_size * 64
                end = min(
                    speech_mix.shape[1], 
                    ((i + 1) * self.chunk_size - 1) * 64 + 512
                )
            else:
                start = i * self.chunk_size * self.frontend.stft.hop_length
                end = min(
                    speech_mix.shape[1], 
                    ((i + 1) * self.chunk_size - 1) * self.frontend.stft.hop_length + self.frontend.stft.win_length
                )
            if (torch.min(speech_mix_lengths) - end) < 1500:
                end = speech_mix.shape[1]
                end_flag = True
            input_speech_mix = speech_mix[:, start:end]
            input_speech_mix_lengths = torch.tensor(
                [max(0, min(end-start, x-start)) for x in speech_mix_lengths],
                dtype=speech_mix_lengths.dtype,
                device=speech_mix_lengths.device,
            )

            start = i * self.chunk_size
            if end_flag:
                end = phn_ref1.shape[1]
            else:
                end = min(
                    phn_ref1.shape[1],
                    (i + 1) * self.chunk_size,
                )
            if (start == end):
                continue
            input_phn_ref1 = phn_ref1[:, start:end]
            input_phn_ref1_lengths = torch.tensor(
                [max(0, min(self.chunk_size, x - i * self.chunk_size)) for x in phn_ref1_lengths],
                dtype=phn_ref1_lengths.dtype,
                device=phn_ref1_lengths.device,
            )
            input_phn_ref2 = phn_ref2[:, start:end]
            input_phn_ref2_lengths = torch.tensor(
                [max(0, min(self.chunk_size, x - i * self.chunk_size)) for x in phn_ref2_lengths],
                dtype=phn_ref2_lengths.dtype,
                device=phn_ref2_lengths.device,
            )

            # 1. Encoder
            encoder_out, encoder_out_lens = self.encode(
                input_speech_mix, input_speech_mix_lengths
            ) # n_spk * (bs, lens, enc_dim)

            ys_hats, ys_hats_lengths, input_phn_ref1, input_phn_ref2 = self._compute_output_layer(
                encoder_out,
                encoder_out_lens,
                input_phn_ref1,
                input_phn_ref2,
                input_phn_ref1_lengths,
                input_phn_ref2_lengths
            )

            targets = [
                input_phn_ref1.contiguous(),
                input_phn_ref2.contiguous(),
            ]
            targets_lengths = [
                input_phn_ref1_lengths,
                input_phn_ref2_lengths,
            ]
            
            loss_ce_perm = torch.stack(
                [
                    self._calc_ce_loss(
                        ys_hats[s_i // self.num_spkrs],
                        ys_hats_lengths[s_i // self.num_spkrs],
                        targets[s_i % self.num_spkrs],
                        targets_lengths[s_i % self.num_spkrs],
                    )
                    for s_i in range(self.num_spkrs ** 2)
                ],
                dim=1
            ) # (bs, n_spk**2)

            loss_ce, min_perm = self.pit.pit_process(loss_ce_perm) # min_perm: bs*[0, 1] or bs*[1, 0]

            all_targets = []
            for s_i in range(self.num_spkrs):
                for n in range(batch_size):
                    all_targets.append(targets[min_perm[n][s_i]][n])
            all_targets = torch.stack(all_targets, dim=0) # (bs*spk, lens)
            all_ys_hats = torch.cat(ys_hats, dim=0)
            acc_ce = th_accuracy(
                all_ys_hats.view(-1, self.vocab_size),
                all_targets,
                ignore_label=self.ignore_id,
            )

            loss.append(loss_ce)
            corr.append(acc_ce * torch.sum(input_phn_ref1_lengths) * 2)
            all_ys_hats_record.append(all_ys_hats)

            if end_flag:
                break

        all_ys_hats_record = torch.cat(all_ys_hats_record, dim=1)
        predictions = all_ys_hats_record.view(-1, self.vocab_size).argmax(-1).cpu().numpy()
        (uniq, counts) = np.unique(predictions, return_counts=True)
        token_variety = uniq.size

        loss = torch.sum(torch.stack(loss))
        acc_ce = torch.sum(torch.stack(corr)) / (torch.sum(phn_ref1_lengths) * 2)
        stats = dict(
            loss=loss.detach(),
            acc=acc_ce,
            token_variety = token_variety,
        )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech_mix: torch.Tensor,
        speech_mix_lengths: torch.Tensor,
        phn_ref1: torch.Tensor,
        phn_ref1_lengths: torch.Tensor,
        phn_ref2: torch.Tensor,
        phn_ref2_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        feats, feats_lengths = self._extract_feats(speech_mix, speech_mix_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        if self.predict_spk:
            encoder_out, encoder_out_lens, encoder_out_spk, encoder_out_lens_spk = self.encoder(feats, feats_lengths)
        else:
            encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)

        if isinstance(encoder_out, list):
            batchsize = speech.size(0)
            for idx, enc_out in enumerate(encoder_out):
                assert enc_out.size(0) == batchsize, (enc_out.size(), batchsize)
                assert enc_out.size(1) <= encoder_out_lens[idx].max(), (
                    enc_out.size(),
                    encoder_out_lens[idx].max(),
                )
        else:
            assert encoder_out.size(0) == speech.size(0), (
                encoder_out.size(),
                speech.size(0),
            )
            assert encoder_out.size(1) <= encoder_out_lens.max(), (
                encoder_out.size(),
                encoder_out_lens.max(),
            )

        if self.predict_spk:
            return encoder_out, encoder_out_lens, encoder_out_spk, encoder_out_lens_spk
        else:
            return encoder_out, encoder_out_lens

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

    def _compute_output_layer(
        self,
        encoder_out,
        encoder_out_lens,
        phn_ref1,
        phn_ref2,
        phn_ref1_lengths,
        phn_ref2_lengths,
    ):
        if isinstance(encoder_out, list):
            if encoder_out[0].shape[1] < phn_ref1.shape[1]:
                phn_ref1 = phn_ref1[:, :encoder_out[0].shape[1]].contiguous()
                phn_ref2 = phn_ref2[:, :encoder_out[0].shape[1]].contiguous()
            elif encoder_out[0].shape[1] > phn_ref1.shape[1]:
                encoder_out[0] = encoder_out[0][:, :phn_ref1.shape[1]].contiguous()
                encoder_out[1] = encoder_out[1][:, :phn_ref1.shape[1]].contiguous()

            if self.cut_begin_end:
                assert encoder_out[0].shape[1]==phn_ref1.shape[1]==phn_ref2.shape[1], (encoder_out[0].shape,phn_ref1.shape,phn_ref2.shape)
                cut_begin = int(0.2 * phn_ref1_lengths.max())
                cut_len = int(0.5 * phn_ref1_lengths.max())
                cut_end = int(0.2 * phn_ref1_lengths.max()) + cut_len
                encoder_out=[enc[:,cut_begin:cut_end] for enc in encoder_out]
                encoder_out_lens= [torch.ones_like(enc) * int(cut_len) for enc in encoder_out_lens]
                targets = [
                    phn_ref1[:, cut_begin : cut_end],
                    phn_ref2[:, cut_begin : cut_end],
                ]
                targets_lengths = encoder_out_lens

            print('encoder_output:',type(encoder_out))
            print('encoder_output:',encoder_out[0].shape)
            ys_hats = [
                self.ce_lo(F.dropout(enc_out, p=self.dropout_rate)) for enc_out in encoder_out
            ] # n_spk * (bs, lens, proj)
            ys_hats_lengths = encoder_out_lens
        else:
            if encoder_out.shape[1] < phn_ref1.shape[1]:
                phn_ref1 = phn_ref1[:, :encoder_out.shape[1]].contiguous()
                phn_ref2 = phn_ref2[:, :encoder_out.shape[1]].contiguous()
            elif encoder_out.shape[1] > phn_ref1.shape[1]:
                encoder_out = encoder_out[:, :phn_ref1.shape[1]].contiguous()

            ys_hats = [
                lo(F.dropout(encoder_out, p=self.dropout_rate)) for lo in self.ce_lo
            ]
            ys_hats_lengths = [encoder_out_lens, encoder_out_lens]
        return ys_hats, ys_hats_lengths, phn_ref1, phn_ref2


    def _calc_ce_loss(
        self,
        ys_hat: torch.Tensor,
        ys_hat_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        bs, seq_len, o_dim = ys_hat.shape
        ys_hat = ys_hat.view(-1, o_dim)
        ys_pad = ys_pad.view(-1)
        loss_ce = self.cross_entropy(ys_hat, ys_pad)  # (batch * seq_len)
        loss_ce = torch.sum(loss_ce.view(bs, seq_len), dim=1)  # (batch)

        return loss_ce

def vq_decode(utt_id, idx_seq, pre_trained_model_root="/data3/VQ_GAN_codebase/egs/vctk/vc1/"):
    """Run decoding process."""
    vq_seq = torch.LongTensor([idx_to_vq[idx] for idx in idx_seq]).to(idx_seq.device)
    assert vq_seq.shape == idx_seq.shape
    print('vq_seq:',vq_seq.shape,vq_seq)
    checkpoint=pre_trained_model_root+"exp/train_nodev_all_vctk_conditioned_melgan_vae.v3/checkpoint-5000000steps.pkl"
    config=default=pre_trained_model_root+"exp/train_nodev_all_vctk_conditioned_melgan_vae.v3/config.yml" 
    verbose=1

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
    device = vq_seq.device
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
        #  torch.LongTensor(vq_seq).view(1, -1).to(device)
        z = vq_seq.long().view(1,-1)
        logging.info(f"Z.shape:", z.shape)
        g = None
        if utt2spk is not None:
            spk_idx = spk2idx[utt2spk[utt_id]]
            g = torch.tensor(spk_idx).long().view(1).to(device)
        g = torch.tensor(3).long().view(1).to(device)
        start = time.time()
        y = model.decode(z, None, g).view(-1).cpu().numpy()
        rtf = (time.time() - start) / (len(y) / config["sampling_rate"])

        # save as PCM 16 bit wav file
        sf.write(os.path.join("tmp_gen/", f"{utt_id}_gen.wav"),
                    y, config["sampling_rate"], "PCM_16")
                    

        # report average RTF
        logging.info(f"Finished generation of utterances {utt_id} (RTF = {rtf:.03f}).")