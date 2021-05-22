#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

min_or_max=max # "min" or "max". This is to determine how the mixtures are generated in local/data.sh.
sample_rate=24k


train_set="tr_${min_or_max}_${sample_rate}"
valid_set="cv_${min_or_max}_${sample_rate}"
test_sets="tt_${min_or_max}_${sample_rate} "

stage=7; stop_stage=100
opts=${opts:-}" --expdir exp"
#asr_conf=train_asr_rnn_1;
#asr_conf=train_asr_mixrnn_2;
asr_conf=train_asr_mixrnn_2_chunk;

# opts=${opts:-}" --expdir exp_center_false"
# asr_conf=train_asr_mixrnn_2_stft_center_false;

# opts=${opts:-}" --expdir exp_melganfrontend"
# asr_conf=train_asr_melganfrontend_mixrnn_2;
# asr_conf=train_asr_melganfrontend_mixrnn_3_chunk;

#lm_conf=train_lm_rnn
lm_conf=train_lm_adam_layers3
lm_infer_conf=decoee_w_kenlm

./hybrid_asr.sh \
    --stage ${stage} --stop_stage ${stop_stage} \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs "${sample_rate}" \
    --ngpu 1 \
    --local_data_opts "--sample_rate ${sample_rate} --min_or_max ${min_or_max}" \
    --asr_config conf/tuning/${asr_conf}.yaml \
    --lm_config conf/${lm_conf}.yaml \
    --lm_infer_config conf/${lm_conf}.yaml \
    --token_type phn \
    --lm_train_text "dump/raw/lm_train.txt" \
    --lm_dev_text "dump/raw/lm_dev.txt" \
    --lm_test_text "dump/raw/lm_test.txt" \
    ${opts} "$@"
