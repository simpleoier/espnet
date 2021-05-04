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

stage=10; stop_stage=10
#asr_conf=train_asr_rnn_1;
asr_conf=train_asr_mixrnn_1;

./hybrid_asr.sh \
    --stage ${stage} --stop_stage ${stop_stage} \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs "${sample_rate}" \
    --ngpu 1 \
    --local_data_opts "--sample_rate ${sample_rate} --min_or_max ${min_or_max}" \
    --asr_config conf/tuning/${asr_conf}.yaml \
    --token_type phn \
    --lm_train_text "data/${train_set}/vq_spk1 data/${train_set}/vq_spk2" \
    "$@"
