home_dir=cv_max_24k
awk '{split($1, lst, "_"); spk=lst[1]"_"lst[4]"_"lst[2]"_"lst[3]"_"lst[5]"_"lst[6]; print(spk, $2)}' ${home_dir}/wav.scp | sort > tmp11 ; mv tmp11 ${home_dir}/wav.scp
awk '{split($1, lst, "_"); spk=lst[1]"_"lst[4]"_"lst[2]"_"lst[3]"_"lst[5]"_"lst[6]; print(spk, $2)}' ${home_dir}/spk1.scp | sort > tmp11 ; mv tmp11 ${home_dir}/spk1.scp
awk '{split($1, lst, "_"); spk=lst[1]"_"lst[4]"_"lst[2]"_"lst[3]"_"lst[5]"_"lst[6]; print(spk, $2)}' ${home_dir}/spk2.scp | sort > tmp11 ; mv tmp11 ${home_dir}/spk2.scp
awk '{split($1, lst, "_"); spk=lst[1]"_"lst[4]"_"lst[2]"_"lst[3]"_"lst[5]"_"lst[6]; print(spk, $2)}' ${home_dir}/utt2spk | sort > tmp11 ; mv tmp11 ${home_dir}/utt2spk
../utils/utt2spk_to_spk2utt.pl ${home_dir}/utt2spk > ${home_dir}/spk2utt

awk '{split($1, lst, "_"); spk=lst[1]"_"lst[4]"_"lst[2]"_"lst[3]"_"lst[5]"_"lst[6]; out=""; for(i=2;i<=NF;i++){out=out" "$i}; print(spk, out)}' ${home_dir}/text_spk1 | sort > tmp11 ; mv tmp11 ${home_dir}/text_spk1
awk '{split($1, lst, "_"); spk=lst[1]"_"lst[4]"_"lst[2]"_"lst[3]"_"lst[5]"_"lst[6]; out=""; for(i=2;i<=NF;i++){out=out" "$i}; print(spk, out)}' ${home_dir}/text_spk2 | sort > tmp11 ; mv tmp11 ${home_dir}/text_spk2
awk '{split($1, lst, "_"); spk=lst[1]"_"lst[4]"_"lst[2]"_"lst[3]"_"lst[5]"_"lst[6]; out=""; for(i=2;i<=NF;i++){out=out" "$i}; print(spk, out)}' ${home_dir}/vq_spk1 | sort > tmp11 ; mv tmp11 ${home_dir}/vq_spk1
awk '{split($1, lst, "_"); spk=lst[1]"_"lst[4]"_"lst[2]"_"lst[3]"_"lst[5]"_"lst[6]; out=""; for(i=2;i<=NF;i++){out=out" "$i}; print(spk, out)}' ${home_dir}/vq_spk2 | sort > tmp11 ; mv tmp11 ${home_dir}/vq_spk2

head ${home_dir}/* -n 1 | awk '{print ($1)}'

