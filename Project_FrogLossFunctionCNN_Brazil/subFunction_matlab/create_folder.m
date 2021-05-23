% create folder
function [] = create_folder(folderM)

if ~exist(folderM, 'dir')
    mkdir(folderM);
end

%[EOF]