% Manual segmentation
%======================%
clc; close all; clear;
%======================%
label_array_org = [ones(1,8), ones(1,5)*2, ones(1,11)*3, ones(1,4)*4, ones(1,4)*5, ones(1,4)*6, ones(1,3)*7, ones(1,5)*8, ones(1,4)*9, ones(1,11)*10];
%======================%
audio_folder = '.\Brazil-Frog\data\';
%======================%
training_data_folder = audio_folder;
training_audio_list = dir(training_data_folder);
training_audio_list = training_audio_list(arrayfun(@(x) ~strcmp(x.name(1), '.'), training_audio_list));
%=====================%
nList = length(training_audio_list);
outResult = cell(1, nList);
for iList = 1:nList
    disp(iList)
    
    speciesName = training_audio_list(iList).name;
    speciesPath = [training_data_folder, '\', speciesName];
    
    [y, fs] = audioread(speciesPath);
    
    y = y-mean(y); % remove mean value
    y = y./max(abs(y)); % normalization
        
    label_data = csvread(['.\Brazil-Frog\label_CSV\', speciesName(1:end-4), '.csv']);
    
    % remove NAN
    nan_index = ~isnan(label_data);
    label_data = label_data(nan_index(:,1),:);
    
    %==============%
    segments = zeros(1, length(y));
    [row, col] = size(label_data);
    for r = 1:row
        temp_start = label_data(r, 1);
        temp_stop = label_data(r, 2);        
        segments(temp_start: temp_stop) = 1;        
    end
       
    % save duration and percentage to cell files
    second(iList) = length(y) / fs;    
    
    frog_1(iList) = sum(segments(segments > 0));
    frog_2(iList) = length(y);
    
    frog_call_percent(iList) = sum(segments(segments > 0)) / length(y);
        
    % %==============%
    % saveFolder = ['.\02_04\manualSyllable\'];
    % create_folder(saveFolder);
    % savePath = [saveFolder, 'fileName.csv'];
    % frog = speciesName(1:(length(speciesName) - 4));
    % csvPath = strrep(savePath, 'fileName', frog);
    % csvwrite(csvPath,segments);
    
end

second_final = zeros(10,1);
frog_call_percent_final = zeros(10,1);
for i = 1:10
    second_final(i) = sum(second(label_array_org == i));
    
    aaa = sum(frog_1(label_array_org == i));
    bbb = sum(frog_2(label_array_org == i));
    frog_call_percent_final(i) = aaa / bbb;
end

second = second_final ;
frog_call_percent = frog_call_percent_final;

%%
figure('Renderer', 'painters', 'Position', [50 50 900 600])
bar(second, 'FaceColor',[0 .5 .5] );

hold on;
len_value = length(second);
value = zeros(1,len_value);
[AX,H1,H2] = plotyy(1:len_value, value, 1:len_value, frog_call_percent);

set(gca, 'XTick', 1:10, ...                             % Change the axes tick marks
    'XTickLabel', {'AAE', 'ATA', 'HMA', 'HCS', 'LFS', 'OOS', 'RGA', 'SRR', 'AHA'}) %   and tick labels

grid on
set(get(AX(2),'Ylabel'),'String','Precentage of frog calls', 'Color', 'g') 
set(H2,'LineWidth',3,'Marker','o', 'Color', 'g');
xtickangle(90);

ylabel('Duration of recording (second)', 'Color', [0 .5 .5] );
xlabel('Frog species');

set(gcf,'color','w');

saveas(gcf, 'duration_percentage_Brazil', 'png')
% export_fig 'duration_percentage_Aus.pdf'

%[EOF]




