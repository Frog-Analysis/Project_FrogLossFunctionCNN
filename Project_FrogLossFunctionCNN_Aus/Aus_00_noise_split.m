% Split data into train and test and label
close all; clear; clc;
%---------------------------------%
% train_percent_array = 0.5:0.1:0.9;
train_percent_array = 0.8;
%---------------------------------%
snrValue_array = [-20:5:-5];
nSnr = length(snrValue_array);
for iSnr = 1:nSnr
    select_snr = snrValue_array(iSnr);
    
    noise_type = { 'white_noise', 'pink_noise', 'rain_noise', 'wind_noise'};
    nNoise = length(noise_type);
    for iNoise = 1:nNoise
        select_noise_type = noise_type{iNoise};
        
        %---------------------------------%
        for i = 1:length(train_percent_array)
            train_percent = train_percent_array(i);
            
            audio_folder = ['.\Australia-Frog\noise_data\', num2str(select_snr), '\noise_data_',  select_noise_type];
            audio_list = dir(audio_folder);
            audio_list = audio_list(arrayfun(@(x) ~strcmp(x.name(1), '.'), audio_list));
            
            % split data into training and testing
            nSpecies = length(audio_list);
            audio_cell = cell(1, nSpecies);
            traing_len_array = zeros(1, nSpecies);
            for iSpecies = 1:nSpecies
                
                speciesName = audio_list(iSpecies).name;
                species_path = [audio_folder, '\', speciesName];
                [audio_data, sr] = audioread(species_path);
                
                audio_length = length(audio_data);
                
                training_len = round(audio_length*train_percent);
                
                training_data = audio_data(1: training_len);
                testing_data = audio_data(training_len+1: audio_length);
                
                traing_len_array(iSpecies) = training_len;
                
                %================%
                training_folder =  ['.\Australia-Frog\data_split_noise\',  'training_', num2str(train_percent), '\', num2str(select_snr), '\noise_data_',  select_noise_type];
                testing_folder = ['.\Australia-Frog\data_split_noise\', 'testing_', num2str(train_percent), '\', num2str(select_snr), '\noise_data_',  select_noise_type];
                
                create_folder(training_folder); create_folder(testing_folder);
                
                training_path = [training_folder, '\', audio_list(iSpecies).name];
                testing_path = [testing_folder, '\', audio_list(iSpecies).name];
                
                audiowrite(training_path,training_data,sr);
                audiowrite(testing_path,testing_data,sr);
                
            end
            
            save_folder_len = '.\Australia-Frog\training_len_folder\';
            create_folder(save_folder_len)
            csvwrite([save_folder_len, 'training_len_', num2str(train_percent), '_noise.csv'], traing_len_array);
            
        end
    end
end
%[EOF]


