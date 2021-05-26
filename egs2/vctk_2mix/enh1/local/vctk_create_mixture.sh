#!/usr/bin/env bash

# Copyright  2021  CASIA (Authors: Jing Shi)
# Apache 2.0

. utils/parse_options.sh

if [ $# -ne 4 ]; then
  echo "Usage: $0 <dir> <vctk-path> <vctk-full-wav> <vctk-2mix-wav>"
  echo " where <dir> is download space,"
  echo " <vctk-path> is the original vctk path"
  echo " <vctk-full-wav> is vctk full wave files path, <vctk-2mix-wav> is wav generation space."
  echo "Note: this script won't actually re-download things if called twice,"
  echo "because we use the --continue flag to 'wget'."
  echo "Note: this script can be used to create vctk_2mix and vctk_2mix corpus"
  echo "Note: <vctk-full-wav> contains all the vctk (or vctk) utterances in wav format,"
  echo "and the directory is organized according to"
  echo "  scripts/mix_2_spk_tr.txt, scripts/mix_2_spk_cv.txt and mix_2_spk_tt.txt"
  echo ", which are the mixture combination schemes."
  exit 1;
fi

dir=$1
vctk_path=$2
vctk_full_wav=$3
vctk_2mix_wav=$4


if ! which matlab >/dev/null 2>&1; then
    echo "matlab not found."
    exit 1
fi

mkdir -p ${dir}
cp local/vctk_mixtures_scripts/*.m ${dir}
cp local/vctk_mixtures_list/*.txt ${dir}

# generate both min and max versions with 8k, 16k and 24k data
sed -i -e "s=/path/to/VCTKWAV_full=${vctk_path}=" \
       -e "s=/path/to/mixtures/2speakers=${vctk_2mix_wav}=" \
       ${dir}/vctk_create_wav_2speakers.m

echo "Creating Mixtures."

matlab_cmd="matlab -nojvm -nodesktop -nodisplay -nosplash -r vctk_create_wav_2speakers"

mixfile=${dir}/mix_matlab.sh
echo "#!/usr/bin/env bash" > $mixfile
echo "cd ${dir}" >> $mixfile
echo $matlab_cmd >> $mixfile
chmod +x $mixfile

# Run Matlab
# (This may take ~6 hours to generate both min and max versions
#  on Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz)
echo "Log is in ${dir}/mix.log"
$train_cmd ${dir}/mix.log $mixfile
