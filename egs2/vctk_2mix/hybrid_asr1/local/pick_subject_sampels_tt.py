import os
import random
import shutil


spk_to_gender={
    "p225": "F", 
    "p226": "M", 
    "p227": "M", 
    "p228": "F", 
    "p229": "F", 
    "p230": "F", 
    "p231": "F", 
    "p232": "M", 
    "p233": "F", 
    "p234": "F", 
    "p236": "F", 
    "p237": "M", 
    "p238": "F", 
    "p239": "F", 
    "p240": "F", 
    "p241": "M", 
    "p243": "M", 
    "p244": "F", 
    "p245": "M", 
    "p246": "M", 
    "p247": "M", 
    "p248": "F", 
    "p249": "F", 
    "p250": "F", 
    "p251": "M", 
    "p252": "M", 
    "p253": "F", 
    "p254": "M", 
    "p255": "M", 
    "p256": "M", 
    "p257": "F", 
    "p258": "M", 
    "p259": "M", 
    "p260": "M", 
    "p261": "F", 
    "p262": "F", 
    "p263": "M", 
    "p264": "F", 
    "p265": "F", 
    "p266": "F", 
    "p267": "F", 
    "p268": "F", 
    "p269": "F", 
    "p270": "M", 
    "p271": "M", 
    "p272": "M", 
    "p273": "M", 
    "p274": "M", 
    "p275": "M", 
    "p276": "F", 
    "p277": "F", 
    "p278": "M", 
    "p279": "M", 
    "p280": "F", 
    "p281": "M", 
    "p282": "F", 
    "p283": "F", 
    "p284": "M", 
    "p285": "M", 
    "p286": "M", 
    "p287": "M", 
    "p288": "F", 
    "p292": "M", 
    "p293": "F", 
    "p294": "F", 
    "p295": "F", 
    "p297": "F", 
    "p298": "M", 
    "p299": "F", 
    "p300": "F", 
    "p301": "F", 
    "p302": "M", 
    "p303": "F", 
    "p304": "M", 
    "p305": "F", 
    "p306": "F", 
    "p307": "F", 
    "p308": "F", 
    "p310": "F", 
    "p311": "M", 
    "p312": "F", 
    "p313": "F", 
    "p314": "F", 
    "p315": "M", 
    "p316": "M", 
    "p317": "F", 
    "p318": "F", 
    "p323": "F", 
    "p326": "M", 
    "p329": "F", 
    "p330": "F", 
    "p333": "F", 
    "p334": "M", 
    "p335": "F", 
    "p336": "F", 
    "p339": "F", 
    "p340": "F", 
    "p341": "F", 
    "p343": "F", 
    "p345": "M", 
    "p347": "M", 
    "p351": "F", 
    "p360": "M", 
    "p361": "F", 
    "p362": "F", 
    "p363": "M", 
    "p364": "M", 
    "p374": "M", 
    "p376": "M", 
    "s5":"F", 
}
father_dir ="/data3/Espnet_xuankai/espnet/egs2/vctk_2mix/hybrid_asr1/"
root_dir = father_dir + "exp/asr_train_asr_conv_dprnn_spk_max_raw_phn/enhanced_tt_max_24k_woLM/"

saved_path = "./samples_for_eval/"

spk1_scp = open(root_dir+'spk1.scp').readlines()
spk2_scp = open(root_dir+'spk2.scp').readlines()

assert len(spk1_scp)==len(spk2_scp)

# tt_selected_combination_2F=["p250_p249","p230_p299","p277_p240",]
# tt_selected_combination_2M=["p311_p363","p336_p334","p360_p262","p341_p364",]
# tt_selected_combination_FM=["p256_p243","p275_p247","p374_p285",]

# tt_selected_combination=tt_selected_combination_2F+tt_selected_combination_2M+tt_selected_combination_FM

list_2f, list_2m, list_fm = [],[],[]
for line in spk1_scp:
    spk, fine_path = line.split(' ')
    spk1, spk2 = spk.split("_")[:2]
    print(spk1,spk2)
    if spk_to_gender[spk1]=="F" and spk_to_gender[spk2]=="F":
        list_2f.append(fine_path.strip())
    elif spk_to_gender[spk1]=="M" and spk_to_gender[spk2]=="M":
        list_2m.append(fine_path.strip())
    else:
        list_fm.append(fine_path.strip())

random.shuffle(list_2f)
random.shuffle(list_2m)
random.shuffle(list_fm)

print(list_2f[:10])
print(list_2m[:10])
print(list_fm[:10])

baseline_root='/data3/Espnet_xuankai/espnet/egs2/vctk_2mix/enh1/exp/enh_train_enh_rnn_tf_raw/'
groundtruth_root='/data3/Espnet/espnet/egs2/vctk_mix/enh1/data/vctk_mix/2speakers/wav24k/max/tt/'
for i in range(50):
    # print(father_dir + list_2f[i])
    # print(saved_path + '/our_method/tt/2females/' + list_2f[i].split("/")[-1])
    shutil.copyfile((father_dir + list_2f[i]).replace("24k","24k_woLM"), saved_path + '/our_method/tt/2females/s1/' + list_2f[i].split("/")[-1] )
    shutil.copyfile((father_dir + list_2f[i]).replace("24k","24k_woLM").replace("wavs/1/","wavs/2/"), saved_path + '/our_method/tt/2females/s2/' + list_2f[i].split("/")[-1] )
    shutil.copyfile(baseline_root+ "/".join(list_2f[i].split("/")[2:]), saved_path + '/baseline_tf/tt/2females/s1/' + list_2f[i].split("/")[-1] )
    shutil.copyfile((baseline_root+"/".join(list_2f[i].split("/")[2:])).replace("wavs/1/","wavs/2/"), saved_path + '/baseline_tf/tt/2females/s2/' + list_2f[i].split("/")[-1] )
    wav_name_reorder=list_2f[i].split("/")[-1].split('_') 
    tmp=wav_name_reorder
    wav_name_reorder='_'.join([tmp[0],tmp[2],tmp[3],tmp[1],tmp[4],tmp[5]])
    shutil.copyfile((groundtruth_root+"/s1/" + wav_name_reorder), saved_path + '/ground_truth/tt/2females/s1/' + list_2f[i].split("/")[-1] )
    shutil.copyfile((groundtruth_root+"/s2/" + wav_name_reorder), saved_path + '/ground_truth/tt/2females/s2/' + list_2f[i].split("/")[-1] )

    shutil.copyfile((father_dir + list_2m[i]).replace("24k","24k_woLM"), saved_path + '/our_method/tt/2males/s1/' + list_2m[i].split("/")[-1] )
    shutil.copyfile((father_dir + list_2m[i]).replace("24k","24k_woLM").replace("wavs/1/","wavs/2/"), saved_path + '/our_method/tt/2males/s2/' + list_2m[i].split("/")[-1] )
    shutil.copyfile(baseline_root+ "/".join(list_2m[i].split("/")[2:]), saved_path + '/baseline_tf/tt/2males/s1/' + list_2m[i].split("/")[-1] )
    shutil.copyfile((baseline_root+"/".join(list_2m[i].split("/")[2:])).replace("wavs/1/","wavs/2/"), saved_path + '/baseline_tf/tt/2males/s2/' + list_2m[i].split("/")[-1] )
    wav_name_reorder=list_2m[i].split("/")[-1].split('_') 
    tmp=wav_name_reorder
    wav_name_reorder='_'.join([tmp[0],tmp[2],tmp[3],tmp[1],tmp[4],tmp[5]])
    shutil.copyfile((groundtruth_root+"/s1/" + wav_name_reorder), saved_path + '/ground_truth/tt/2males/s1/' + list_2m[i].split("/")[-1] )
    shutil.copyfile((groundtruth_root+"/s2/" + wav_name_reorder), saved_path + '/ground_truth/tt/2males/s2/' + list_2m[i].split("/")[-1] )

    shutil.copyfile((father_dir + list_fm[i]).replace("24k","24k_woLM"), saved_path + '/our_method/tt/1f1m/s1/' + list_fm[i].split("/")[-1] )
    shutil.copyfile((father_dir + list_fm[i]).replace("24k","24k_woLM").replace("wavs/1/","wavs/2/"), saved_path + '/our_method/tt/1f1m/s2/' + list_fm[i].split("/")[-1] )
    shutil.copyfile(baseline_root+ "/".join(list_fm[i].split("/")[2:]), saved_path + '/baseline_tf/tt/1f1m/s1/' + list_fm[i].split("/")[-1] )
    shutil.copyfile((baseline_root+"/".join(list_fm[i].split("/")[2:])).replace("wavs/1/","wavs/2/"), saved_path + '/baseline_tf/tt/1f1m/s2/' + list_fm[i].split("/")[-1] )
    wav_name_reorder=list_fm[i].split("/")[-1].split('_') 
    tmp=wav_name_reorder
    wav_name_reorder='_'.join([tmp[0],tmp[2],tmp[3],tmp[1],tmp[4],tmp[5]])
    shutil.copyfile((groundtruth_root+"/s1/" + wav_name_reorder), saved_path + '/ground_truth/tt/1f1m/s1/' + list_fm[i].split("/")[-1] )
    shutil.copyfile((groundtruth_root+"/s2/" + wav_name_reorder), saved_path + '/ground_truth/tt/1f1m/s2/' + list_fm[i].split("/")[-1] )





    



