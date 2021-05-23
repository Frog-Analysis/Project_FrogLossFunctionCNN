% Manual segmentation
%======================%
clc; close all; clear;
%======================%
audio_folder = '.\Australia-Frog\data\';
%======================%
training_data_folder = audio_folder;
training_audio_list = dir(training_data_folder);
training_audio_list = training_audio_list(arrayfun(@(x) ~strcmp(x.name(1), '.'), training_audio_list));
%=====================%
nList = length(training_audio_list);
outResult = cell(1, nList);
for iList = 1:nList
    
    speciesName = training_audio_list(iList).name;
    speciesPath = [training_data_folder, '\', speciesName];
    
    [y, fs] = audioread(speciesPath);
    
    y = y-mean(y); % remove mean value
    y = y./max(abs(y)); % normalization
        
    label_data = xlsread(['.\Australia-Frog\label\', speciesName(1:end-4), '.xlsx']);
    
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
       
    % 
    second(iList) = length(y) / fs;    
    frog_call_percent(iList) = sum(segments(segments > 0)) / length(y);
        
    % %==============%
    % saveFolder = ['.\02_04\manualSyllable\'];
    % create_folder(saveFolder);
    % savePath = [saveFolder, 'fileName.csv'];
    % frog = speciesName(1:(length(speciesName) - 4));
    % csvPath = strrep(savePath, 'fileName', frog);
    % csvwrite(csvPath,segments);
    
end
%-----------------------%
%%
figure('Renderer', 'painters', 'Position', [50 50 900 600])
bar(second, 'FaceColor',[0 .5 .5] );

hold on;
len_value = length(second);
value = zeros(1,len_value);
[AX,H1,H2] = plotyy(1:len_value, value, 1:len_value, frog_call_percent);

set(gca, 'XTick', 1:23, ...                             % Change the axes tick marks
    'XTickLabel', {'ADI', 'CPA', 'CSA', 'LTS', 'LTE','LCA', 'LCS', 'LFX', 'LGA', 'LLA', 'LNA', ...
    'LRA', 'LTI', 'LVI','MFS','MFI','NSI','PKN','PCA','PRI','RSS','UFA', 'ULA'}) %   and tick labels

grid on
set(get(AX(2),'Ylabel'),'String','Precentage of frog calls', 'Color', 'g') 
set(H2,'LineWidth',3,'Marker','o', 'Color', 'g');
xtickangle(90);

ylabel('Duration of recording (second)', 'Color', [0 .5 .5] );
xlabel('Frog species');

set(gcf,'color','w');

saveas(gcf, 'duration_percentage_Aus', 'png')
% export_fig 'duration_percentage_Aus.pdf'

%[EOF]




