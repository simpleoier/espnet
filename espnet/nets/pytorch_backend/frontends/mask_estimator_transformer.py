from typing import Tuple

import numpy as np
import torch
from torch.nn import functional as F
from torch_complex.tensor import ComplexTensor

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttentionTimeRestricted
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayerTimeRestricted
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import MultiLayeredConv1d
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import PositionwiseFeedForward
from espnet.nets.pytorch_backend.transformer.repeat import repeat


class MaskEstimator(torch.nn.Module):
    def __init__(self, type, idim, 
                 attention_dim=256,
                 attention_heads=4,
                 linear_units=2048,
                 num_blocks=6,
                 dropout_rate=0.1,
                 positional_dropout_rate=0.1,
                 attention_dropout_rate=0.0,
                 pos_enc_class=PositionalEncoding,
                 normalize_before=False,
                 concat_after=False,
                 positionwise_layer_type="linear",
                 positionwise_conv_kernel_size=1,
                 padding_idx=-1,
                 nmask=1,
                 att_restr_window=15):
        super(MaskEstimator, self).__init__()

        self.linear_transform = torch.nn.Linear(idim, attention_dim)
        self.normalize_before = normalize_before
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (attention_dim, linear_units, dropout_rate)
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (attention_dim, linear_units, positionwise_conv_kernel_size, dropout_rate)
        else:
            raise NotImplementedError("Support only linear or conv1d.")
        self.masknet = repeat(
            num_blocks,
            lambda: EncoderLayerTimeRestricted(
                attention_dim,
                MultiHeadedAttentionTimeRestricted(attention_heads, attention_dim, attention_dropout_rate),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                normalize_before,
                concat_after,
                time_window=att_restr_window,
            )
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)

        self.nmask = nmask
        self.linears = torch.nn.ModuleList(
            [torch.nn.Linear(attention_dim, idim) for _ in range(nmask)])

        self.window_size = att_restr_window
        pad_front = int(self.window_size / 2)
        pad_end = self.window_size - pad_front - 1
        self.mask_pad = (pad_front, pad_end)  # pad the last dimension by (pad_front, pad_end)

    def forward(self, xs: ComplexTensor, ilens: torch.LongTensor) \
            -> Tuple[Tuple[torch.Tensor, ...], torch.LongTensor]:
        """The forward function

        Args:
            xs: (B, F, C, T)
            ilens: (B,)
        Returns:
            hs (torch.Tensor): The hidden vector (B, F, C, T)
            masks: A tuple of the masks. (B, F, C, T)
            ilens: (B,)
        """
        assert xs.size(0) == ilens.size(0), (xs.size(0), ilens.size(0))
        bs, freq, C, input_length = xs.size()
        # (B, F, C, T) -> (B, C, T, F)
        xs = xs.permute(0, 2, 3, 1)

        # Calculate amplitude: (B, C, T, F) -> (B, C, T, F)
        xs = (xs.real ** 2 + xs.imag ** 2) ** 0.5
        # xs: (B, C, T, F) -> xs: (B * C, T, F)
        xs = xs.view(-1, xs.size(-2), xs.size(-1))
        # ilens: (B,) -> ilens_: (B * C)
        ilens_ = ilens[:, None].expand(-1, C).contiguous().view(-1)

        src_mask = (~make_pad_mask(ilens_.tolist())).to(xs.device).unsqueeze(-2)  # (B*C, 1, maxlen)
        src_mask = F.pad(src_mask, self.mask_pad, "constant", 0).unfold(-1, self.window_size, 1)

        # xs: (B * C, T, F) -> xs: (B * C, T, D)
        xs = self.linear_transform(xs)
        xs, _ = self.masknet(xs, src_mask)
        # xs: (B * C, T, D) -> xs: (B, C, T, D)
        xs = xs.view(-1, C, xs.size(-2), xs.size(-1))

        masks = []  # Final mask for speeches and noises
        for linear in self.linears:
            # xs: (B, C, T, D) -> mask:(B, C, T, F)
            mask = linear(xs)

            mask = torch.sigmoid(mask)
            # Zero padding
            mask.masked_fill(make_pad_mask(ilens, mask, length_dim=2), 0)

            # (B, C, T, F) -> (B, F, C, T)
            mask = mask.permute(0, 3, 1, 2)

            # Take cares of multi gpu cases: If input_length > max(ilens)
            if mask.size(-1) < input_length:
                mask = F.pad(mask, [0, input_length - mask.size(-1)], value=0)
            masks.append(mask)

        return tuple(masks), ilens
