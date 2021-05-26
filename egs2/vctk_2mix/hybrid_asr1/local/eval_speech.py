import speechmetrics
from pathlib import Path
from tqdm import tqdm
import numpy as np

my_metrics = speechmetrics.load('absolute')
file_path = "/data3/Espnet_xuankai/espnet/egs2/vctk_2mix/hybrid_asr1_min/exp/asr_train_asr_mixrnn_spk_raw_phn/enhanced_cv_min_24k_raw/" #logdir/output.1/wavs/1/"

file_path = "/data3/Espnet_xuankai/espnet/egs2/vctk_2mix/hybrid_asr1_min/exp/asr_train_asr_mixrnn_spk_raw_phn/enhanced_tt_min_24k/" #logdir/output.1/wavs/1/"

#file_path = "/data3/Espnet_xuankai/espnet/egs2/vctk_2mix/enh1/exp/enh_train_enh_rnn_tf_raw/enhanced_cv_max_24k" #logdir/output.1/wavs/1/"
#file_path = "/data3/Espnet_xuankai/espnet/egs2/vctk_2mix/enh1/exp/enh_train_enh_rnn_tf_raw/enhanced_tt_max_24k" #logdir/output.1/wavs/1/"
#
#file_path = "/data3/Espnet_xuankai/espnet/egs2/vctk_2mix/hybrid_asr1/exp/asr_train_asr_conv_dprnn_spk_max_raw_phn/enhanced_cv_max_24k_woLM/" #logdir/output.1/wavs/1/"
file_path = "/data3/Espnet_xuankai/espnet/egs2/vctk_2mix/hybrid_asr1/exp/asr_train_asr_conv_dprnn_spk_max_raw_phn/enhanced_cv_max_24k/" #logdir/output.1/wavs/1/"

file_path = "/data3/Espnet_xuankai/espnet/egs2/vctk_2mix/hybrid_asr1/exp/asr_train_asr_conv_dprnn_spk_max_raw_phn/enhanced_cv_max_24k_rnnlm/" #logdir/output.1/wavs/1/"

wav_path = Path(file_path)
# scores = my_metrics(file_path+"p227_p252_397_1.0791_370_-1.0791.wav")

wav_list = wav_path.rglob("*.wav")
mos_score, srmr_score = np.array([]),np.array([])
for wav in tqdm(wav_list): 
    print(str(wav))
    try:
        score = my_metrics(str(wav))
    except:
        continue
    mos_score=np.append(mos_score,score['mosnet'])
    srmr_score=np.append(srmr_score,score['srmr'])
    print(score)
    print(mos_score.mean())

    print(srmr_score.mean())
