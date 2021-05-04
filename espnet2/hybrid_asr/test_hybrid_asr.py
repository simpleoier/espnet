#!/usr/bin/python
import pdb

import numpy as np
import torch

from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.hybrid_asr.espnet_model import ESPnetHybridASRModel
from espnet2.hybrid_asr.cross_entropy import CrossEntropy


def build_model():
    vocab_size = 5
    token_list = ['a', 'e', 'i', 'o', 'u']
    frontend = DefaultFrontend()
    specaug = None
    normalize = None
    preencoder = None
    decoder = None
    rnnt_decoder = None

    encoder = TransformerEncoder(
        input_size=80,
        output_size=256,
        attention_heads=4,
        linear_units=512,
        num_blocks=1,
        input_layer='linear',
    )
    # cross_entropy = CrossEntropy(
    #     odim=5,
    #     encoder_output_sizse=256,
    # )

    hybrid_model = ESPnetHybridASRModel(
        vocab_size=vocab_size,
        frontend=frontend,
        specaug=specaug,
        normalize=normalize,
        preencoder=preencoder,
        encoder=encoder,
        decoder=decoder,
        cross_entropy=None,
        rnnt_decoder=rnnt_decoder,
        token_list=token_list,
    )
    return hybrid_model


def generate_random_data():
    batch_size = 3
    # Input
    speech_lens = np.random.randint(1, 4, batch_size) * 1600
    speech = np.random.rand(batch_size, np.max(speech_lens))
    target_lens = speech_lens // 128 + 1
    target = np.random.randint(0, 5, size=(batch_size, max(target_lens)))

    for i in range(batch_size):
        speech[i][speech_lens[i]:] = 0
        target[i][target_lens[i]:] = -1

    return (
        torch.from_numpy(speech).float(),
        torch.from_numpy(speech_lens).long(),
        torch.from_numpy(target).long(),
        torch.from_numpy(target_lens).long(),
    )

def main():
    hybrid_model = build_model()
    speech, speech_lens, target, target_lens = generate_random_data()
    pdb.set_trace()
    loss, stats, weight = hybrid_model(speech, speech_lens, target, target_lens)


if __name__=="__main__":
    main()