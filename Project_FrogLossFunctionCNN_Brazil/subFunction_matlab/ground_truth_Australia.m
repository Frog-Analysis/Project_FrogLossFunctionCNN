function [label_data] = ground_truth_Australia(label_path, ~, ~)

% read label
label_data = xlsread(label_path);

% remove NAN
nan_index = ~isnan(label_data);
label_data = label_data(nan_index(:,1),:);
label_data = sortrows(label_data,2);

end