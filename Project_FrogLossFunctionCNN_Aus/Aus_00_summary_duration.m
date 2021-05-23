
clc; close all; clear;

base_folder = 'F:\project_code\Project_FrogLossFunctionCNN_Aus\Australia-Frog\data_split';
train_audio_list = folder_to_list([base_folder, '\training_0.8']);

nList = length(train_audio_list);
train_duration = zeros(1, nList);
for iList = 1:nList
    
    speciesName = train_audio_list(iList).name;
    speciesPath = [base_folder, '\training_0.8', '\', speciesName];
    
    [y, fs] = audioread(speciesPath);
    train_duration(iList) = length(y) / fs;
    
end

test_audio_list = folder_to_list([base_folder, '\testing_0.8']);
nList = length(test_audio_list);
test_duration = zeros(1, nList);
for iList = 1:nList
    
    speciesName = test_audio_list(iList).name;
    speciesPath = [base_folder, '\testing_0.8', '\', speciesName];
    
    [y, fs] = audioread(speciesPath);
    test_duration(iList) = length(y) / fs;
    
end


disp(sum(train_duration))
disp(sum(test_duration))






