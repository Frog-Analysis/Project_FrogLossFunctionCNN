% Obtain list from folder
function [my_list, cnt_list] = folder_to_list(my_folder)

my_list = dir(my_folder);
my_list = my_list(arrayfun(@(x) ~strcmp(x.name(1), '.'), my_list));

cnt_list = length(my_list);

end
%[EOF]


