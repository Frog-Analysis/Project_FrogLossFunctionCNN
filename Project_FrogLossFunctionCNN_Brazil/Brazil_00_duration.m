% Aus and Brazil frog call duration
% labeling
close all; clear; clc;

fs = 44100;

%% Brazil data
disp('Start Brazil data analysis')
label_array_org = [ones(1,8), ones(1,5)*2, ones(1,11)*3, ones(1,4)*4, ones(1,4)*5, ones(1,4)*6, ones(1,3)*7, ones(1,5)*8, ones(1,4)*9, ones(1,11)*10];

data_folder = '.\Brazil-Frog\data\';
audioList = dir(data_folder);
audioList = audioList(arrayfun(@(x) ~strcmp(x.name(1), '.'), audioList));
nAudio = length(audioList);
audio_duration = zeros(nAudio, 1);
for iAudio = 1:nAudio
    disp(audioList(iAudio).name);
    
    audio_path = [data_folder, audioList(iAudio).name];
    [y, fs] =  audioread(audio_path);
    audio_duration(iAudio) = length(y) / fs;
    
end

% labeling
label_folder = '.\Brazil-Frog\label\';
label_list = dir(label_folder);
label_list = label_list(arrayfun(@(x) ~strcmp(x.name(1), '.'), label_list));
nTrainingList = length(label_list);
duration_array = zeros(nTrainingList, 1);
% mean_std_array = zeros(nTrainingList, 2);
len_cell = cell(nTrainingList, 1);
for iTrainingList = 1:nTrainingList
    frog_name = label_list(iTrainingList).name;
    frog_name = frog_name(1:end-4);
    label_path = [label_folder, label_list(iTrainingList).name];
    
    label_data = ground_truth_Brazil(label_path, frog_name, fs);
    
    start_timestamp = label_data(:, 1);
    stop_timestamp = label_data(:, 2);
    len = stop_timestamp - start_timestamp + 1;
    duration_tmp = sum(len);
    duration_array(iTrainingList) = duration_tmp/fs;
    
    len_cell{iTrainingList} = len;

end


for i = 1:10
    select_idx = label_array_org == i;
    
    select_call = duration_array(select_idx);
    select_all = audio_duration(select_idx);
    call_sum(i) = sum(select_call);
    all_sum(i) = sum(select_all);
    
    len_tmp = len_cell(select_idx);
    len_mat = cell2mat(len_tmp);
    
    mean_value(i) = mean(len_mat);
    std_value(i) = std(len_mat);
    
end

percent_brazil = call_sum ./ all_sum;

days = 1:length(percent_brazil);

figure;
yyaxis left
b = bar(days, all_sum);
b.FaceColor = [ 0 0.847 0.841 ];
yyaxis right
p = plot(days, percent_brazil, '-gs', 'LineWidth', 2, 'Color', 'k', 'MarkerEdgeColor','b',...
    'MarkerFaceColor',[0.5,0.5,0.5]);

xlabel('Frog species')
yyaxis left
ylabel('Duration of recording (second)')
yyaxis right
ylabel('Percentage of frog call', 'color', 'k')

x_name = {'AAE','ATA','AHA','HMA','HCS','HCE','LFS','OOS','RGA','SRR'};
xticks(days)
xticklabels(x_name);
xtickangle(45)

% export_fig duration_percentage_brazil.pdf



figure;
errorbar(mean_value/fs, std_value/fs,'-s','MarkerSize',10,...
    'MarkerEdgeColor','red','MarkerFaceColor','red','linewidth',2);
ylim([-0.5, 7])
x_name = {'AAE','ATA','AHA','HMA','HCS','HCE','LFS','OOS','RGA','SRR'};
xticks(days)
xticklabels(x_name);
xtickangle(45)
ylabel('Duration of call (second)')
grid on;






