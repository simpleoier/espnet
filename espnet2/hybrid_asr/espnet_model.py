from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

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


def pick_longer_utt(ref1,ref2):
    """ ref1,ref2: shape of [bs, label_lens]"""
    longer_ref = []
    for s1,s2 in zip(ref1,ref2):
        s1_mask =  (s1 == 0).int() 
        s2_mask =  (s2 == 0).int()
        if s1_mask[-30:].sum() > 20:
            longer_ref.append(s1)
        elif s2_mask[-30:].sum() > 20:
            longer_ref.append(s2)
        else:
            print("s1 s2 last 30:", s1[-30:], s2[-30:])
            raise ValueError
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
    ):
        assert check_argument_types()
        assert rnnt_decoder is None, "Not implemented"

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.token_list = token_list.copy()

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
        self.cross_entropy = torch.nn.CrossEntropyLoss(
            weight=torch.FloatTensor(normedWeights),
            ignore_index=ignore_id,
            reduce=False,
            reduction="none"
        )
        self.dropout_rate = 0.0
        self.cut_begin_end = False
        self.only_longer_ref = False

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
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            target: (Batch, Length)
            target_lengths: (Batch,)
        """
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
            longer_phn_ref = pick_longer_utt(phn_ref1,phn_ref2)
            target = longer_phn_ref
            targets_lengths =  phn_ref1_lengths

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech_mix, speech_mix_lengths) # n_spk * (bs, lens, enc_dim)
        ys_hats, ys_hats_lengths = self._compute_output_layer(encoder_out, encoder_out_lens, phn_ref1, phn_ref2)
        if self.only_longer_ref:
            loss_ce = self._calc_ce_loss(ys_hats[0],ys_hats_lengths[0],targets,targets_lengths)
            all_ys_hats = ys_hats[0]
            all_targets = targets
            assert all_ys_hats.shape == all_targets.shape
            acc_ce = th_accuracy(
                all_ys_hats,
                all_targets,
                ignore_label=self.ignore_id,
            )
            print("ys_hats:", ys_hats[0].shape, ys_hats[0][0].max(-1)[1][800:820])
            print("targets:", all_targets[0][800:820].data.cpu().numpy())
            print("predict:", all_ys_hats.max(-1)[1][0][800:820].data.cpu().numpy(), '\n')
            loss = loss_ce

            stats = dict(
                loss=loss.detach(),
                acc=acc_ce,
            )

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

        loss = loss_ce

        stats = dict(
            loss=loss.detach(),
            acc=acc_ce,
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
        encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)

        if isinstance(encoder_out, list):
            batchsize = speech.size(0)
            for i, enc_out in enumerate(encoder_out):
                assert enc_out.size(0) == batchsize, (enc_out.size(), batchsize)
                assert enc_out.size(1) <= encoder_out_lens[i].max(), (
                    enc_out.size(),
                    encoder_out_lens[i].max(),
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
                assert encoder_out[0].shape[1]==phn_ref1.shape[1]==phn_ref2.shape[1], (encoder_out[0].shape,phn_ref1.shape,phn_ref2.shapie)
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
        return ys_hats, ys_hats_lengths


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
