% Change label from txt to csv
close all; clear; clc;
%======================%
label_array_org = [ones(1,8), ones(1,5)*2, ones(1,11)*3, ones(1,4)*4, ones(1,4)*5, ones(1,4)*6, ones(1,3)*7, ones(1,5)*8, ones(1,4)*9, ones(1,11)*10];
%======================%
fs = 44100;
audio_folder = ['.\Brazil-Frog\data_split\'];
%======================%
win_size_array = [0.2, 0.5, 1] * fs;
win_over_array = [0.8];
nSize = length(win_size_array);
nOver = length(win_over_array);
for iSize = 1:nSize
    win_size = win_size_array(iSize);
    for iOver = 1:nOver
        win_over = win_over_array(iOver);
        percent_array = 0.8;
        %percent_array = 0.5:0.1:0.9;
        nPerc = length(percent_array);
        for iPerc = 1:nPerc
            temp_percent = percent_array(iPerc);
            training_data_folder = [audio_folder, 'training_', num2str(temp_percent)];
            training_audio_list = dir(training_data_folder);
            training_audio_list = training_audio_list(arrayfun(@(x) ~strcmp(x.name(1), '.'), training_audio_list));
            
            testing_data_folder = [audio_folder, 'testing_', num2str(temp_percent)];
            testing_audio_list = dir(testing_data_folder);
            testing_audio_list = testing_audio_list(arrayfun(@(x) ~strcmp(x.name(1), '.'), testing_audio_list));
            
            % start label location
            training_len_array = csvread(['.\Brazil-Frog\training_len_folder\training_len_', num2str(temp_percent), '.csv']);
            
            % labeling
            label_folder = '.\Brazil-Frog\label\';
            label_list = dir(label_folder);
            label_list = label_list(arrayfun(@(x) ~strcmp(x.name(1), '.'), label_list));
            nTrainingList = length(label_list);
            for iTrainingList = 1:nTrainingList
                frog_name = label_list(iTrainingList).name;
                frog_name = frog_name(1:end-4);
                label_path = [label_folder, label_list(iTrainingList).name];
                
                label_data = ground_truth_Brazil(label_path, frog_name, fs);
                        
                % save data
                save_final_folder = ['.\Brazil-Frog\label_CSV\'];
                create_folder(save_final_folder);
                csvwrite([save_final_folder, ['\', frog_name, '.csv']], label_data);
                
            end
        end
    end
end
%[EOF]

