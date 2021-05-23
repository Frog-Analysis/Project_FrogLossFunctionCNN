% Extract raw noisy data and split it into training and testing data
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
                
                % find the closest value
                start_label = training_len_array(iTrainingList);
                label_array = sort(label_data(:));
                [minValue, closestIndex] = min(abs(start_label-label_array));
                
                if closestIndex == length(label_array)
                    disp('To do')
                    training_label_data = label_data;
                    testing_label_data = [];
                else
                    [training_label_data, testing_label_data] = label_split(label_data, label_array, closestIndex, start_label);
                end
                
                % training data
                tra_species_path = [training_data_folder, '\',  training_audio_list(iTrainingList).name];
                [training_audio_signal, ~] = audioread(tra_species_path);
                
                training_label_info = ones(length(training_audio_signal), 1);
                [nRow, ~] = size(training_label_data);
                for iRow = 1:nRow
                    start = training_label_data(iRow, 1);
                    stop = min(training_label_data(iRow, 2), length(training_label_info));
                    training_label_info(start:stop) = 2;
                end
                training_label_info = training_label_info - 1;
                
                % generate label for each sliding window
                training_label_mat = window_move(training_label_info, win_size, win_over);
                training_final_label = label_mat_to_array(training_label_mat, win_size, label_array_org(iTrainingList));
                training_final_label = training_final_label(1,:);
                
                % windowing
                training_audio_feature = window_move(training_audio_signal, win_size, win_over);
                
                % combine feature and label
                training_temp_data = [training_audio_feature', training_final_label'];
                
                % testing data
                tes_species_path = [testing_data_folder, '\',  testing_audio_list(iTrainingList).name];
                [testing_audio_signal, ~] = audioread(tes_species_path);
                
                % labeling
                testing_label_info = ones(length(testing_audio_signal), 1) ;
                [nRow, ~] = size(testing_label_data);
                if nRow == 0
                    testing_label_info = testing_label_info - 1;
                else
                    for iRow = 1:nRow
                        start = max(testing_label_data(iRow, 1), 1);
                        stop = testing_label_data(iRow, 2);
                        testing_label_info(start: stop) = 2;
                    end
                    testing_label_info = testing_label_info - 1;
                end
                
                % generate label for each sliding window
                testing_label_mat = window_move(testing_label_info, win_size, win_over);
                testing_final_label = label_mat_to_array(testing_label_mat, win_size, label_array_org(iTrainingList));
                testing_final_label = testing_final_label(1,:);
                
                % windowing
                [testing_audio_feature, ~] = window_move(testing_audio_signal, win_size, win_over);
                
                % combine feature and label
                testing_temp_data = [testing_audio_feature', testing_final_label'];
                
                % save data
                save_final_folder = ['.\Brazil-Frog\0329_raw_data_clean\percent_', num2str(temp_percent), '_winsize_', num2str(win_size), '_winover_',  num2str(win_over), '\', frog_name];
                create_folder(save_final_folder);
                save_data_training = training_temp_data;
                csvwrite([save_final_folder, '\train.csv'], save_data_training);
                
                my_len_testing = min(size(testing_temp_data, 1));
                save_data_testing = testing_temp_data;
                csvwrite([save_final_folder, '\test.csv'], save_data_testing);
                
            end
        end
    end
end
%[EOF]

