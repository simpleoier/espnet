#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail



train_set=tr05_multi_noisy_si84 # tr05_multi_noisy (original training data) or tr05_multi_noisy_si284 (add si284 data)
valid_set=dt05_multi_isolated_1ch_track
test_sets="et05_simu_isolated_1ch_track"  # "\
# dt05_real_isolated_1ch_track dt05_simu_isolated_1ch_track et05_real_isolated_1ch_track et05_simu_isolated_1ch_track \
# dt05_real_beamformit_2mics dt05_simu_beamformit_2mics et05_real_beamformit_2mics et05_simu_beamformit_2mics \
# dt05_real_beamformit_5mics dt05_simu_beamformit_5mics et05_real_beamformit_5mics et05_simu_beamformit_5mics \
# "

#asr_config=conf/train_asr_rnn.yaml
#asr_config=conf/train_asr_conformer6_n_fft400_hop_length160.yaml
asr_config=conf/tuning/train_asr_conv_dprnn_spk_max.yaml
inference_config=conf/decode_asr_rnn.yaml
lm_config=conf/train_lm.yaml


use_word_lm=false

./hybrid_asr.sh                                   \
    --train_set "${train_set}"             \
    --valid_set "${valid_set}"             \
    --test_sets "${test_sets}"             \
    --lang en \
    --token_type phn                      \
    --feats_type raw               \
    --feats_normalize null         \
    --asr_config "${asr_config}"           \
    --inference_config "${inference_config}"     \
    --lm_config "${lm_config}"             \
    --use_word_lm ${use_word_lm}           \
    --lm_train_text "data/${train_set}/vq_text" \
    --lm_dev_text "data/${valid_set}/vq_text" \
    --lm_test_text "data/${test_sets}/vq_text" \
    ${opts:-} "$@"
