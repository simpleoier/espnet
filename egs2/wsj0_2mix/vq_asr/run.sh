#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_si284
valid_set=test_dev93
test_sets="test_dev93 test_eval92"

min_or_max=max # "min" or "max". This is to determine how the mixtures are generated in local/data.sh.
sample_rate=8k


train_set="tr_${min_or_max}_${sample_rate}"
valid_set="cv_${min_or_max}_${sample_rate}"
test_sets="tt_${min_or_max}_${sample_rate} "

stage=9; stop_stage=13
opts=${opts:-}" --expdir exp"

./vq_asr.sh \
    --stage ${stage} --stop_stage ${stop_stage} \
    --use_lm true \
    --token_type char \
    --nbpe 80 \
    --nlsyms_txt data/nlsyms.txt \
    --lm_config conf/train_lm_transformer.yaml \
    --asr_config conf/tuning/vq_asr_transformer.yaml \
    --inference_config conf/decode.yaml \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs "${sample_rate}" \
    --feats_normalize false \
    --ngpu 4 \
    --bpe_train_text "data/train_si284/text" \
    --lm_train_text "data/train_si284/text data/local/other_text/text" "$@"
