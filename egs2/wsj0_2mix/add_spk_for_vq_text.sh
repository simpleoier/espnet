dsfrate=32
vq_dir='wsj0_8k_dsf32/exp/train_nodev_all_spk_8k_vctk_conditioned_melgan_vae.v3.finetune/wav/checkpoint-6000000steps/'
aim_dir='vq_asr/data/'

cat ${vq_dir}/max_cv_s1_unnorm_8k/text | awk '{split($1, lst, "_"); spk=substr(lst[1],1,3)"_"substr(lst[3],1,3); print(spk"_"$0)}' | sort > ${aim_dir}/cv_max_8k/vq_spk1_dsf${dsfrate}
cat ${vq_dir}/max_cv_s2_unnorm_8k/text | awk '{split($1, lst, "_"); spk=substr(lst[1],1,3)"_"substr(lst[3],1,3); print(spk"_"$0)}' | sort > ${aim_dir}/cv_max_8k/vq_spk2_dsf${dsfrate}
cat ${vq_dir}/max_tr_s2_unnorm_8k/text | awk '{split($1, lst, "_"); spk=substr(lst[1],1,3)"_"substr(lst[3],1,3); print(spk"_"$0)}' | sort > ${aim_dir}/tr_max_8k/vq_spk2_dsf${dsfrate}
cat ${vq_dir}/max_tr_s1_unnorm_8k/text | awk '{split($1, lst, "_"); spk=substr(lst[1],1,3)"_"substr(lst[3],1,3); print(spk"_"$0)}' | sort > ${aim_dir}/tr_max_8k/vq_spk1_dsf${dsfrate}
cat ${vq_dir}/max_tt_s1_unnorm_8k/text | awk '{split($1, lst, "_"); spk=substr(lst[1],1,3)"_"substr(lst[3],1,3); print(spk"_"$0)}' | sort > ${aim_dir}/tt_max_8k/vq_spk1_dsf${dsfrate}
cat ${vq_dir}/max_tt_s2_unnorm_8k/text | awk '{split($1, lst, "_"); spk=substr(lst[1],1,3)"_"substr(lst[3],1,3); print(spk"_"$0)}' | sort > ${aim_dir}/tt_max_8k/vq_spk2_dsf${dsfrate}

cat ${vq_dir}/min_tt_s2_unnorm_8k/text | awk '{split($1, lst, "_"); spk=substr(lst[1],1,3)"_"substr(lst[3],1,3); print(spk"_"$0)}' | sort > ${aim_dir}/tt_min_8k/vq_spk2_dsf${dsfrate}
cat ${vq_dir}/min_tt_s1_unnorm_8k/text | awk '{split($1, lst, "_"); spk=substr(lst[1],1,3)"_"substr(lst[3],1,3); print(spk"_"$0)}' | sort > ${aim_dir}/tt_min_8k/vq_spk1_dsf${dsfrate}
cat ${vq_dir}/min_cv_s1_unnorm_8k/text | awk '{split($1, lst, "_"); spk=substr(lst[1],1,3)"_"substr(lst[3],1,3); print(spk"_"$0)}' | sort > ${aim_dir}/cv_min_8k/vq_spk1_dsf${dsfrate}
cat ${vq_dir}/min_cv_s2_unnorm_8k/text | awk '{split($1, lst, "_"); spk=substr(lst[1],1,3)"_"substr(lst[3],1,3); print(spk"_"$0)}' | sort > ${aim_dir}/cv_min_8k/vq_spk2_dsf${dsfrate}
cat ${vq_dir}/min_tr_s2_unnorm_8k/text | awk '{split($1, lst, "_"); spk=substr(lst[1],1,3)"_"substr(lst[3],1,3); print(spk"_"$0)}' | sort > ${aim_dir}/tr_min_8k/vq_spk2_dsf${dsfrate}
cat ${vq_dir}/min_tr_s1_unnorm_8k/text | awk '{split($1, lst, "_"); spk=substr(lst[1],1,3)"_"substr(lst[3],1,3); print(spk"_"$0)}' | sort > ${aim_dir}/tr_min_8k/vq_spk1_dsf${dsfrate}

