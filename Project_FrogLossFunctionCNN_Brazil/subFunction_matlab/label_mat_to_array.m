function [final_label] = label_mat_to_array(label_mat, win_size, assign_label)

[~, nCol] = size(label_mat);
final_label_weak = zeros(1, nCol);
final_label_mid = zeros(1, nCol);
final_label_strong = zeros(1, nCol);
for iCol = 1:nCol
    temp_label = label_mat(:, iCol);
    
    % percentage
    if sum(temp_label) / win_size == 1
        final_label_strong(iCol) = assign_label;
    end
    
    if sum(temp_label) / win_size >= 0.75
        final_label_mid(iCol) = assign_label;
    end
    
    if sum(temp_label) / win_size >= 0.5
        final_label_weak(iCol) = assign_label;
    end
end

final_label = [final_label_weak; final_label_mid; final_label_strong];

end

