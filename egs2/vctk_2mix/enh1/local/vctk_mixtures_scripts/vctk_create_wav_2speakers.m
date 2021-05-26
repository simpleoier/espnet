% vctk_create_wav_2_speakers.m
%
% Create 2-speaker mixtures
% 
% This script assumes that VCTK (v0.80) using the original folder structure
% under VCTK-Corpus/, e.g., 
% /path/to/VCTKWAV_FULL/wav48/p279/p279_350.wav.
%
% The original script was created by the MERL in Deep Clustering for WSJ0. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Copyright (C) 2021  CASIA (Jing Shi)
%   Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


data_type = {'tr','cv','tt'};
vctkroot = '/path/to/VCTKWAV_full/'; % YOUR_PATH/, the folder containing VCTK/
output_dir24k='/path/to/mixtures/2speakers/wav24k';
output_dir16k='/path/to/mixtures/2speakers/wav16k';
output_dir8k='/path/to/mixtures/2speakers/wav8k';

min_max = {'min','max'};

useaudioread = 0;
if exist('audioread','file')
    useaudioread = 1;
end

for i_mm = 1:length(min_max)
    for i_type = 1:length(data_type)
        if ~exist([output_dir24k '/' min_max{i_mm} '/' data_type{i_type}],'dir')
            mkdir([output_dir24k '/' min_max{i_mm} '/' data_type{i_type}]);
        end
        if ~exist([output_dir16k '/' min_max{i_mm} '/' data_type{i_type}],'dir')
            mkdir([output_dir16k '/' min_max{i_mm} '/' data_type{i_type}]);
        end
        if ~exist([output_dir8k '/' min_max{i_mm} '/' data_type{i_type}],'dir')
            mkdir([output_dir8k '/' min_max{i_mm} '/' data_type{i_type}]);
        end
        status = mkdir([output_dir8k  '/' min_max{i_mm} '/' data_type{i_type} '/s1/']); %#ok<NASGU>
        status = mkdir([output_dir8k  '/' min_max{i_mm} '/' data_type{i_type} '/s2/']); %#ok<NASGU>
        status = mkdir([output_dir8k  '/' min_max{i_mm} '/' data_type{i_type} '/mix/']); %#ok<NASGU>
        status = mkdir([output_dir16k '/' min_max{i_mm} '/' data_type{i_type} '/s1/']); %#ok<NASGU>
        status = mkdir([output_dir16k '/' min_max{i_mm} '/' data_type{i_type} '/s2/']); %#ok<NASGU>
        status = mkdir([output_dir16k '/' min_max{i_mm} '/' data_type{i_type} '/mix/']); %#ok<NASGU>
        status = mkdir([output_dir24k '/' min_max{i_mm} '/' data_type{i_type} '/s1/']); %#ok<NASGU>
        status = mkdir([output_dir24k '/' min_max{i_mm} '/' data_type{i_type} '/s2/']); %#ok<NASGU>
        status = mkdir([output_dir24k '/' min_max{i_mm} '/' data_type{i_type} '/mix/']);
                
        TaskFile = ['vctk_mix_2_spk_' data_type{i_type} '.txt'];
        fid=fopen(TaskFile,'r');
        C=textscan(fid,'%s %f %s %f');
        
        Source1File = ['vctk_mix_2_spk_' min_max{i_mm} '_' data_type{i_type} '_1'];
        Source2File = ['vctk_mix_2_spk_' min_max{i_mm} '_' data_type{i_type} '_2'];
        MixFile     = ['vctk_mix_2_spk_' min_max{i_mm} '_' data_type{i_type} '_mix'];
        
        fid_s1 = fopen(Source1File,'w');
        fid_s2 = fopen(Source2File,'w');
        fid_m  = fopen(MixFile,'w');
        
        num_files = length(C{1});
        fs8k=8000;
        fs16k=16000;
        fs24k=24000;
        
        scaling_24k = zeros(num_files,2);
        scaling_16k = zeros(num_files,2);
        scaling_8k = zeros(num_files,2);
        scaling16bit_24k = zeros(num_files,1);
        scaling16bit_16k = zeros(num_files,1);
        scaling16bit_8k = zeros(num_files,1);
        fprintf(1,'%s\n',[min_max{i_mm} '_' data_type{i_type}]);
        for i = 1:num_files
            [inwav1_dir,invwav1_name,inwav1_ext] = fileparts(C{1}{i});
            [inwav2_dir,invwav2_name,inwav2_ext] = fileparts(C{3}{i});
            fprintf(fid_s1,'%s\n',C{1}{i});
            fprintf(fid_s2,'%s\n',C{3}{i});
            inwav1_snr = C{2}(i);
            inwav2_snr = C{4}(i);
            mix_name = [invwav1_name,'_',num2str(inwav1_snr),'_',invwav2_name,'_',num2str(inwav2_snr)];
            fprintf(fid_m,'%s\n',mix_name);
                        
            % get input wavs
            if useaudioread
                [s1, fs] = audioread([vctkroot C{1}{i}]);
                s2       = audioread([vctkroot C{3}{i}]);
            else                
                [s1, fs] = wavread([vctkroot C{1}{i}]); %#ok<*DWVRD>
                s2       = wavread([vctkroot C{3}{i}]);            
            end
            
            % resample, normalize 8 kHz file, save scaling factor
            s1_8k=resample(s1,fs8k,fs);
            [s1_8k,lev1]=activlev(s1_8k,fs8k,'n'); % y_norm = y /sqrt(lev);
            s2_8k=resample(s2,fs8k,fs);
            [s2_8k,lev2]=activlev(s2_8k,fs8k,'n');
                        
            weight_1=10^(inwav1_snr/20);
            weight_2=10^(inwav2_snr/20);
            
            s1_8k = weight_1 * s1_8k;
            s2_8k = weight_2 * s2_8k;
            
            switch min_max{i_mm}
                case 'max'
                    mix_8k_length = max(length(s1_8k),length(s2_8k));
                    s1_8k = cat(1,s1_8k,zeros(mix_8k_length - length(s1_8k),1));
                    s2_8k = cat(1,s2_8k,zeros(mix_8k_length - length(s2_8k),1));
                case 'min'
                    mix_8k_length = min(length(s1_8k),length(s2_8k));
                    s1_8k = s1_8k(1:mix_8k_length);
                    s2_8k = s2_8k(1:mix_8k_length);
            end
            mix_8k = s1_8k + s2_8k;
                    
            max_amp_8k = max(cat(1,abs(mix_8k(:)),abs(s1_8k(:)),abs(s2_8k(:))));
            mix_scaling_8k = 1/max_amp_8k*0.9;
            s1_8k = mix_scaling_8k * s1_8k;
            s2_8k = mix_scaling_8k * s2_8k;
            mix_8k = mix_scaling_8k * mix_8k;
            
            % apply same gain to 16 kHz file
            s1_16k=resample(s1,fs16k,fs);
            s2_16k=resample(s2,fs16k,fs);

            s1_16k = weight_1 * s1_16k / sqrt(lev1);
            s2_16k = weight_2 * s2_16k / sqrt(lev2);
            
            switch min_max{i_mm}
                case 'max'
                    mix_16k_length = max(length(s1_16k),length(s2_16k));
                    s1_16k = cat(1,s1_16k,zeros(mix_16k_length - length(s1_16k),1));
                    s2_16k = cat(1,s2_16k,zeros(mix_16k_length - length(s2_16k),1));
                case 'min'
                    mix_16k_length = min(length(s1_16k),length(s2_16k));
                    s1_16k = s1_16k(1:mix_16k_length);
                    s2_16k = s2_16k(1:mix_16k_length);
            end
            mix_16k = s1_16k + s2_16k;
            
            max_amp_16k = max(cat(1,abs(mix_16k(:)),abs(s1_16k(:)),abs(s2_16k(:))));
            mix_scaling_16k = 1/max_amp_16k*0.9;
            s1_16k = mix_scaling_16k * s1_16k;
            s2_16k = mix_scaling_16k * s2_16k;
            mix_16k = mix_scaling_16k * mix_16k;            

            % apply same gain to 24 kHz file
            s1_24k=resample(s1,fs24k,fs);
            s2_24k=resample(s2,fs24k,fs);
            s1_24k = weight_1 * s1_24k / sqrt(lev1);
            s2_24k = weight_2 * s2_24k / sqrt(lev2);
            
            switch min_max{i_mm}
                case 'max'
                    mix_24k_length = max(length(s1_24k),length(s2_24k));
                    s1_24k = cat(1,s1_24k,zeros(mix_24k_length - length(s1_24k),1));
                    s2_24k = cat(1,s2_24k,zeros(mix_24k_length - length(s2_24k),1));
                case 'min'
                    mix_24k_length = min(length(s1_24k),length(s2_24k));
                    s1_24k = s1_24k(1:mix_24k_length);
                    s2_24k = s2_24k(1:mix_24k_length);
            end
            mix_24k = s1_24k + s2_24k;
            
            max_amp_24k = max(cat(1,abs(mix_24k(:)),abs(s1_24k(:)),abs(s2_24k(:))));
            mix_scaling_24k = 1/max_amp_24k*0.9;
            s1_24k = mix_scaling_24k * s1_24k;
            s2_24k = mix_scaling_24k * s2_24k;
            mix_24k = mix_scaling_24k * mix_24k;            
            
            % save 8 kHz and 16 kHz mixtures, as well as
            % necessary scaling factors
            
            scaling_24k(i,1) = weight_1 * mix_scaling_24k/ sqrt(lev1);
            scaling_24k(i,2) = weight_2 * mix_scaling_24k/ sqrt(lev2);
            scaling_16k(i,1) = weight_1 * mix_scaling_16k/ sqrt(lev1);
            scaling_16k(i,2) = weight_2 * mix_scaling_16k/ sqrt(lev2);
            scaling_8k(i,1) = weight_1 * mix_scaling_8k/ sqrt(lev1);
            scaling_8k(i,2) = weight_2 * mix_scaling_8k/ sqrt(lev2);
            
            scaling16bit_24k(i) = mix_scaling_24k;
            scaling16bit_16k(i) = mix_scaling_16k;
            scaling16bit_8k(i)  = mix_scaling_8k;
            
            if useaudioread                          
                s1_8k = int16(round((2^15)*s1_8k));
                s2_8k = int16(round((2^15)*s2_8k));
                mix_8k = int16(round((2^15)*mix_8k));
                s1_16k = int16(round((2^15)*s1_16k));
                s2_16k = int16(round((2^15)*s2_16k));
                mix_16k = int16(round((2^15)*mix_16k));
                s1_24k = int16(round((2^15)*s1_24k));
                s2_24k = int16(round((2^15)*s2_24k));
                mix_24k = int16(round((2^15)*mix_24k));
                audiowrite([output_dir8k '/' min_max{i_mm} '/' data_type{i_type} '/s1/' mix_name '.wav'],s1_8k,fs8k);
                audiowrite([output_dir16k '/' min_max{i_mm} '/' data_type{i_type} '/s1/' mix_name '.wav'],s1_16k,fs16k);
                audiowrite([output_dir24k '/' min_max{i_mm} '/' data_type{i_type} '/s1/' mix_name '.wav'],s1_24k,fs24k);
                audiowrite([output_dir8k '/' min_max{i_mm} '/' data_type{i_type} '/s2/' mix_name '.wav'],s2_8k,fs8k);
                audiowrite([output_dir16k '/' min_max{i_mm} '/' data_type{i_type} '/s2/' mix_name '.wav'],s2_16k,fs16k);
                audiowrite([output_dir24k '/' min_max{i_mm} '/' data_type{i_type} '/s2/' mix_name '.wav'],s2_24k,fs24k);
                audiowrite([output_dir8k '/' min_max{i_mm} '/' data_type{i_type} '/mix/' mix_name '.wav'],mix_8k,fs8k);
                audiowrite([output_dir16k '/' min_max{i_mm} '/' data_type{i_type} '/mix/' mix_name '.wav'],mix_16k,fs16k);
                audiowrite([output_dir24k '/' min_max{i_mm} '/' data_type{i_type} '/mix/' mix_name '.wav'],mix_24k,fs24k);
            else
                wavwrite(s1_8k,fs8k,[output_dir8k '/' min_max{i_mm} '/' data_type{i_type} '/s1/' mix_name '.wav']); %#ok<*DWVWR>
                wavwrite(s1_16k,fs16k,[output_dir16k '/' min_max{i_mm} '/' data_type{i_type} '/s1/' mix_name '.wav']);
                wavwrite(s1_24k,fs24k,[output_dir24k '/' min_max{i_mm} '/' data_type{i_type} '/s1/' mix_name '.wav']);
                wavwrite(s2_8k,fs8k,[output_dir8k '/' min_max{i_mm} '/' data_type{i_type} '/s2/' mix_name '.wav']);
                wavwrite(s2_16k,fs16k,[output_dir16k '/' min_max{i_mm} '/' data_type{i_type} '/s2/' mix_name '.wav']);
                wavwrite(s2_24k,fs24k,[output_dir24k '/' min_max{i_mm} '/' data_type{i_type} '/s2/' mix_name '.wav']);
                wavwrite(mix_8k,fs8k,[output_dir8k '/' min_max{i_mm} '/' data_type{i_type} '/mix/' mix_name '.wav']);
                wavwrite(mix_16k,fs16k,[output_dir16k '/' min_max{i_mm} '/' data_type{i_type} '/mix/' mix_name '.wav']);
                wavwrite(mix_24k,fs24k,[output_dir24k '/' min_max{i_mm} '/' data_type{i_type} '/mix/' mix_name '.wav']);
            end
            
            if mod(i,10)==0
                fprintf(1,'.');
                if mod(i,200)==0
                    fprintf(1,'\n');
                end
            end
            
        end
        save([output_dir8k  '/' min_max{i_mm} '/' data_type{i_type} '/scaling.mat'],'scaling_8k','scaling16bit_8k');
        save([output_dir16k '/' min_max{i_mm} '/' data_type{i_type} '/scaling.mat'],'scaling_16k','scaling16bit_16k');
        save([output_dir24k '/' min_max{i_mm} '/' data_type{i_type} '/scaling.mat'],'scaling_24k','scaling16bit_24k');
        
        fclose(fid);
        fclose(fid_s1);
        fclose(fid_s2);
        fclose(fid_m);
    end
end
