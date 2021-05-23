function [training_label_data, testing_label_data] = label_split(label_data, label_array, closestIndex, start_label)

if label_array(closestIndex) > start_label
    
    start_index = closestIndex - 1;
    stop_index = closestIndex;
    
    start_value = label_array(start_index);
    stop_value = label_array(stop_index);
    
    start_loc = label_data == start_value;
    start_sum_loc = sum(start_loc);
    stop_loc = label_data == stop_value;
    stop_sum_loc = sum(stop_loc);
    
    if start_sum_loc(2) == 1 && stop_sum_loc(1) ==1
        training_label_data = label_data(1: ceil(start_index/2), :);
        testing_label_data = label_data((ceil(start_index/2)+1):end, :) - start_label + 1;
        
    else
        % create new
        new_label_array = [label_array(1:closestIndex-1); start_label; start_label;label_array(closestIndex:end)];
        new_label_data = reshape(new_label_array, 2, length(new_label_array)/2)';
        
        training_label_data = new_label_data(1: ceil(start_index/2), :);
        testing_label_data = new_label_data((ceil(start_index/2)+1):end, :) - start_label + 1;
    end
    
else
    start_index = closestIndex;
    stop_index = closestIndex + 1;
    
    start_value = label_array(start_index);
    stop_value = label_array(stop_index);
    
    start_loc = label_data == start_value;
    start_sum_loc = sum(start_loc);
    stop_loc = label_data == stop_value;
    stop_sum_loc = sum(stop_loc);
    
    if start_sum_loc(2) == 1 && stop_sum_loc(1) ==1
        
        training_label_data = label_data(1: ceil(start_index/2), :);
        testing_label_data = label_data((ceil(start_index/2)+1):end, :) - start_label +1;
        
    else
        % create new
        if closestIndex + 3 > length(label_array)
            new_label_array = [label_array(1:closestIndex); start_label; start_label; label_array(end)];
        else
            new_label_array = [label_array(1:closestIndex); start_label; start_label; label_array(closestIndex+3:end)];
        end
        new_label_data = reshape(new_label_array,  2, length(new_label_array)/2)';
        
        training_label_data = new_label_data(1: ceil(start_index/2), :);
        testing_label_data = new_label_data((ceil(start_index/2)+1):end, :) - start_label + 1;
    end
end

end

