% Apply a sliding window to X and segment it into windowed signal
function [res, res_loc] = window_move(signal, win_size, win_over)

[row_value, col_value] = size(signal);

if row_value < col_value
    signal = signal';
end

len_signal = length(signal);
num_signal = floor((len_signal - win_size*win_over) / (win_size*(1-win_over)));
win_signal = cell(1, num_signal);
res_start_loc = zeros(num_signal, 1);
res_stop_loc = zeros(num_signal, 1);
for idxSignal = 1:num_signal
    start_signal = int64((idxSignal - 1)*win_size*(1-win_over)+1);
    stop_signal = start_signal+win_size-1;
    temp_sig = signal(start_signal:stop_signal);
    win_signal{idxSignal} = temp_sig;
    
    res_start_loc(idxSignal) = start_signal;
    res_stop_loc(idxSignal) = stop_signal;
    
end

res = cell2mat(win_signal);
res_loc = [res_start_loc, res_stop_loc];

end
%[EOF]
