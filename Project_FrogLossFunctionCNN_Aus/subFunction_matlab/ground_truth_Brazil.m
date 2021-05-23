function [result] = ground_truth_Brazil(label_path, frog_name, fs)

fileID = fopen(label_path);
C = textscan(fileID,'%s %s %s %s' );
fclose(fileID);

start_array = C{1,1};
stop_array = C{1,2};
idx = 1;
start_timestamp = [];
stop_timestamp = [];
if (strcmp(frog_name, 'HypsiboasCordobae_3') || strcmp(frog_name, 'HypsiboasCordobae_4') ...
        || strcmp(frog_name, 'OsteocephalusOophagus_1')  || strcmp(frog_name, 'HypsiboasCinerascens_2')...
        || strcmp(frog_name, 'HypsiboasCinerascens_3') || strcmp(frog_name, 'HypsiboasCinerascens_4') ...
        || strcmp(frog_name, 'HylaMinuta_7') || strcmp(frog_name, 'HylaMinuta_8') || strcmp(frog_name, 'HylaMinuta_9') ...
        || strcmp(frog_name, 'HypsiboasCinerascens_3') || strcmp(frog_name, 'HypsiboasCinerascens_4'))
    for k = 1:length(start_array)
        start_time = start_array{k};
        stop_time = stop_array{k};
        start_timestamp(idx) = round(str2double(start_time) * fs );
        stop_timestamp(idx) = round(str2double(stop_time) * fs );
        idx = idx + 1;
    end
else
    for k = 1:2:length(start_array)
        start_time = start_array{k};
        stop_time = stop_array{k};
        start_timestamp(idx) = round(str2double(start_time) * fs );
        stop_timestamp(idx) = round(str2double(stop_time) * fs );
        idx = idx + 1;
    end
end


% combine start and stop timestamps
label_data = [start_timestamp', stop_timestamp'];

% remove NAN
nan_index = ~isnan(label_data);
label_data = label_data(nan_index(:,1),:);
result = sortrows(label_data,2);

end

