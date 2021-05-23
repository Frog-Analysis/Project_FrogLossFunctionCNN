% Split data into train and test and label
close all; clear; clc;
%---------------------------------%
% train_percent_array = 0.5:0.1:0.9;
train_percent_array = 0.5:0.1:0.8;

%---------------------------------%
for i = 1:length(train_percent_array)
    train_percent = train_percent_array(i);
    
    audio_folder = ['.\Brazil-Frog\data\'];
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
        training_folder =  ['.\Brazil-Frog\data_split\',  'training_', num2str(train_percent)];
        testing_folder = ['.\Brazil-Frog\data_split\', 'testing_', num2str(train_percent)];
        
        create_folder(training_folder); create_folder(testing_folder);
        
        training_path = [training_folder, '\', audio_list(iSpecies).name];
        testing_path = [testing_folder, '\', audio_list(iSpecies).name];
        
        audiowrite(training_path,training_data,sr);
        audiowrite(testing_path,testing_data,sr);
        
    end
    
    save_folder_len = '.\Brazil-Frog\training_len_folder\';
    create_folder(save_folder_len)
    csvwrite([save_folder_len, 'training_len_', num2str(train_percent), '.csv'], traing_len_array);
    
end
%[EOF]


