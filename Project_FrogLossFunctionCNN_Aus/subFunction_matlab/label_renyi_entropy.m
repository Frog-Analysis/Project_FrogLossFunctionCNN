function [training_signal_label_entropy, testing_signal_label_entropy] = label_renyi_entropy(species_folder, frog_name, start_label, win_size_1, win_over_1)

%% Use entropy for segmentation
% windowing
species_path = [species_folder, frog_name];
[audio_signal, fs] = audioread(species_path);
win_size_2 = round(0.01*fs);
win_over_2 = 0;
[audio_feature, ~] = window_move(audio_signal, win_size_2, win_over_2);

nCol = size(audio_feature, 2);
renyi_array = zeros(nCol, 1);
for iCol = 1:nCol
    win_signal = audio_feature(:,iCol);
    renyi_array(iCol) = abs(renyi_entro(win_signal, 3));
end

E = renyi_array;
E = (E - min(E))./(max(E) - min(E));

block_size = 50;
nBlock = ceil(length(E) / block_size);
block_thresh_small = zeros(1, nBlock);
block_thresh_mid = zeros(1, nBlock);
block_thresh_big = zeros(1, nBlock);
for iBlock = 1: nBlock
    start = (iBlock-1)*block_size +1;
    stop = start + block_size - 1;
    
    block_E = E(start:min(stop, length(E)));
    [idx, value] = hist(block_E, 10);
    [~,thresh_loc] = max(idx);
    
    block_thresh_small(iBlock) = value(min(max(thresh_loc-1, 1), length(value)));
    block_thresh_mid(iBlock) = value(min(thresh_loc, length(value)));
    block_thresh_big(iBlock) = value(min(thresh_loc+1, length(value)));
end

if length(block_thresh_small) == 1
    sample_thresh_small = block_thresh_small;
else
    block_thresh_small = [block_thresh_small(1:end-1), block_thresh_small(end-1)];
    sample_thresh_small = repelem(block_thresh_small, block_size);
end

if length(block_thresh_mid) == 1
    sample_thresh_mid = block_thresh_mid;
else
    block_thresh_mid = [block_thresh_mid(1:end-1), block_thresh_mid(end-1)];
    sample_thresh_mid = repelem(block_thresh_mid, block_size);
end

if length(block_thresh_big) == 1
    sample_thresh_big = block_thresh_big;
else
    block_thresh_big = [block_thresh_big(1:end-1), block_thresh_big(end-1)];
    sample_thresh_big = repelem(block_thresh_big, block_size);
end

% return back
audio_signal_len = length(audio_signal);
signal_label_small = ones(audio_signal_len,1);
signal_label_mid = ones(audio_signal_len,1);
signal_label_big = ones(audio_signal_len,1);
for iBlock = 1:nBlock   
    start = (iBlock - 1) * win_size_2 * block_size + 1;
    stop = start + win_size_2 * block_size;
    
    if E(iBlock) < sample_thresh_small(iBlock)
        signal_label_small(start:stop) = 0;
    end
    if E(iBlock) < sample_thresh_mid(iBlock)
        signal_label_mid(start:stop) = 0;
    end
    if E(iBlock) < sample_thresh_big(iBlock)
        signal_label_big(start:stop) = 0;
    end
end

% label to all
signal_label_small = signal_label_small(1:audio_signal_len);
training_signal_label_small = signal_label_small(1:start_label);
training_signal_label_small_mat = window_move(training_signal_label_small, win_size_1, win_over_1);
training_signal_label_entropy = label_mat_to_array(training_signal_label_small_mat, win_size_1, 1);
training_signal_label_entropy = training_signal_label_entropy(1,:);


testing_singal_label_small = signal_label_small(start_label+1:end);
testing_signal_label_small_mat = window_move(testing_singal_label_small,  win_size_1, win_over_1);
testing_signal_label_entropy = label_mat_to_array(testing_signal_label_small_mat, win_size_1, 1);
testing_signal_label_entropy = testing_signal_label_entropy(1,:);



end

