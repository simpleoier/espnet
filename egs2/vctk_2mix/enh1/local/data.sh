#!/usr/bin/env bash

# Copyright 2020  CASIA (Authors: Jing Shi)
# Apache 2.0
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

help_message=$(cat << EOF
Usage: $0 [--min_or_max <min/max>] [--sample_rate <8k/16k/24k>]
  optional argument:
    [--min_or_max]: min, max
    [--sample_rate]: 8k , 16k, 24k
EOF
)

. ./db.sh

vctk_full_wav=$PWD/data/vctk/vctk_wav
vctk_2mix_wav=$PWD/data/vctk_mix/2speakers
vctk_2mix_scripts=$PWD/data/vctk_mix/scripts


other_text=data/local/other_text/text
nlsyms=data/nlsyms.txt
min_or_max=min
sample_rate=24k


. utils/parse_options.sh

if [ $# -ne 0 ]; then
    echo "${help_message}"
    exit 1;
fi

if [ ! -e "${VCTK}" ]; then
    log "Fill the value of 'vctk' of db.sh"
    exit 1
fi

train_set="tr_"${min_or_max}_${sample_rate}
train_dev="cv_"${min_or_max}_${sample_rate}
recog_set="tt_"${min_or_max}_${sample_rate}

### This part is for vctk mix
### Create mixtures for 2 speakers
if [ ! -d ${vctk_2mix_wav} ]; then
    local/vctk_create_mixture.sh ${vctk_2mix_scripts} ${VCTK} ${vctk_full_wav} \
        ${vctk_2mix_wav} || exit 1;
else
    log "Already exists. Skipped."
fi

local/vctk_2mix_data_prep.sh --min-or-max ${min_or_max} --sample-rate ${sample_rate} \
    ${vctk_2mix_wav}/wav${sample_rate}/${min_or_max} ${vctk_2mix_scripts} ${VCTK} || exit 1;

### create .scp file for reference audio
for folder in ${train_set} ${train_dev} ${recog_set};
do
    sed -e 's/\/mix\//\/s1\//g' ./data/$folder/wav.scp > ./data/$folder/spk1.scp
    sed -e 's/\/mix\//\/s2\//g' ./data/$folder/wav.scp > ./data/$folder/spk2.scp
done


### Also need vctk corpus to prepare language information
### This is from Kaldi vctk recipe
log "local/vctk_data_prep.sh ${vctk}/??-{?,??}.? ${vctk1}/??-{?,??}.?"
local/vctk_data_prep.sh ${vctk}/??-{?,??}.? ${vctk1}/??-{?,??}.?
log "local/vctk_format_data.sh"
local/vctk_format_data.sh
log "mkdir -p data/vctk"
mkdir -p data/vctk
log "mv data/{dev_dt_*,local,test_dev*,test_eval*,train_si284} data/vctk"
mv data/{dev_dt_*,local,test_dev*,test_eval*,train_si284} data/vctk