#!/usr/bin/env bash

# Copyright  2021  CASIA (Authors: Jing Shi)
# Apache 2.0
set -e
set -u
set -o pipefail

min_or_max=min
sample_rate=8k

. utils/parse_options.sh
. ./path.sh

if [ $# -le 2 ]; then
  echo "Arguments should be VCTK-2MIX directory, the mixing script path and the VCTK path, see ../run.sh for example."
  exit 1;
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

normalize_transcript=$KALDI_ROOT/egs/wsj/s5/local/normalize_transcript.pl

wavdir=$1
srcdir=$2
vctk_full_wav=$3

tr="tr_${min_or_max}_${sample_rate}"
cv="cv_${min_or_max}_${sample_rate}"
tt="tt_${min_or_max}_${sample_rate}"

# check if the wav dir exists.
for f in $wavdir/tr $wavdir/cv $wavdir/tt; do
  if [ ! -d $wavdir ]; then
    echo "Error: $wavdir is not a directory."
    exit 1;
  fi
done

# check if the script file exists.
for f in $srcdir/vctk_mix_2_spk_${min_or_max}_tr_mix $srcdir/vctk_mix_2_spk_${min_or_max}_cv_mix $srcdir/vctk_mix_2_spk_${min_or_max}_tt_mix; do
  if [ ! -f $f ]; then
    echo "Could not find $f.";
    exit 1;
  fi
done

data=./data

for x in tr cv tt; do
  target_folder=$(eval echo \$$x)
  mkdir -p ${data}/$target_folder
  cat $srcdir/vctk_mix_2_spk_${min_or_max}_${x}_mix | \
    awk -v dir=$wavdir/$x '{printf("%s %s/mix/%s.wav\n", $1, dir, $1)}' | sort > ${data}/${target_folder}/wav.scp
    awk '{split($1, lst, "_"); spk=lst[1]"_"lst[4]"_"lst[2]"_"lst[3]"_"lst[5]"_"lst[6]; print(spk, $2)}' ${data}/${target_folder}/wav.scp | \
     sort > tmp_wav.scp 
  mv tmp_wav.scp ${data}/${target_folder}/wav.scp 

  awk '{split($1, lst, "_"); spk=lst[1]"_"lst[2]; print($1, spk)}' ${data}/${target_folder}/wav.scp | sort > ${data}/${target_folder}/utt2spk
  utt2spk_to_spk2utt.pl ${data}/${target_folder}/utt2spk > ${data}/${target_folder}/spk2utt
done

# transcriptions (only for 'max' version)
if [[ "$min_or_max" = "min" ]]; then
  exit 0
fi

for x in tr cv tt; do
  target_folder=$(eval echo \$$x)

  awk '{split($1, lst, "_"); spk=lst[1]; utt=lst[1]"_"lst[3]; print(spk"/"utt".txt")}' ${data}/${target_folder}/wav.scp > ${data}/${target_folder}/utt_spk1
  awk -v dir=${vctk_full_wav} '{printf("%s/txt/%s\n", dir, $1)}' ${data}/${target_folder}/utt_spk1 > ${data}/${target_folder}/tmp_spk1_0
  cat ${data}/${target_folder}/tmp_spk1_0 | while read line 
  do
      cat ${line} | sed 's/`	//g' # avoid the bad line of """It'`    s the tip of the iceberg."""
      [ `tail -n1 ${line} | wc -l` -eq 1 ] || echo ""  # add newline if there is no "\n" in text
  done > ${data}/${target_folder}/tmp_spk1_1
  
  awk '{split($1, lst, "_"); spk=lst[2]; utt=lst[2]"_"lst[5]; print(spk"/"utt".txt")}' ${data}/${target_folder}/wav.scp  > ${data}/${target_folder}/utt_spk2
  awk -v dir=${vctk_full_wav} '{printf("%s/txt/%s\n", dir, $1)}' ${data}/${target_folder}/utt_spk2 > ${data}/${target_folder}/tmp_spk2_0
  cat ${data}/${target_folder}/tmp_spk2_0 | while read line 
  do
      cat ${line} | sed 's/`	//g' # avoid the bad line of """It'`    s the tip of the iceberg."""
      [ `tail -n1 ${line} | wc -l` -eq 1 ] || echo ""  # add newline if there is no "\n" in text
  done > ${data}/${target_folder}/tmp_spk2_1

  # Do some basic normalization steps.  At this point we don't remove OOVs--
  # that will be done inside the training scripts, as we'd like to make the
  # data-preparation stage independent of the specific lexicon used.
  # TODO: Need to normalize it.
  noiseword="<NOISE>"
  cat ${data}/${target_folder}/tmp_spk1_1 | ${normalize_transcript} ${noiseword}  > ${data}/${target_folder}/tmp_spk1_2|| exit 1;
  cat ${data}/${target_folder}/tmp_spk2_1 | ${normalize_transcript} ${noiseword}  > ${data}/${target_folder}/tmp_spk2_2|| exit 1;

  paste -d" " ${data}/${target_folder}/wav.scp ${data}/${target_folder}/tmp_spk1_2 | awk '{$2=""; print($0)}' > ${data}/${target_folder}/text_spk1
  paste -d" " ${data}/${target_folder}/wav.scp ${data}/${target_folder}/tmp_spk2_2 | awk '{$2=""; print($0)}' > ${data}/${target_folder}/text_spk2

  rm ${data}/${target_folder}/tmp*
  rm ${data}/${target_folder}/utt_spk*
done
