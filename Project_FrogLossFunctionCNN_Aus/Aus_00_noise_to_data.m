% add background noise
clc; close all; clear;
%=============%
baseFolder = '.\Australia-Frog\data';
%=============%
wavin3_pink = load('.\noise\pink.mat'); % load babble noise
wavin3_pink = wavin3_pink.pink;

[wavin3_rain, ~] = audioread('.\noise\rain.wav');
wavin3_rain = wavin3_rain(:, 1);

wavin3_babble = load('.\noise\babble.mat'); % load babble noise
wavin3_babble = wavin3_babble.babble;

[wavin3_wind, ~] = audioread('.\noise\wind.wav');
wavin3_wind = wavin3_wind(:, 1);

%=============%
% snrValueArray = [-20, -10, 0, 10, 20];
snrValueArray = [-5, -10, -15, -20];
nSNR = length(snrValueArray);
for iSNR = 1:nSNR
    
    snrValue = snrValueArray(iSNR);
    
    speciesList = dir(baseFolder);
    speciesList = speciesList(arrayfun(@(x) ~strcmp(x.name(1), '.'), speciesList));
    nSpecies = length(speciesList);
    for iSpecies = 1:nSpecies
        
        species_path = [baseFolder, '\', speciesList(iSpecies).name];
        [cleanSignal, fs] = audioread(species_path);
        
        % save folder
        save_folder = '.\Australia-Frog\noise_data\';
        
        % add white noise
        noisedSignal_white = awgn(cleanSignal, snrValue, 'measured');
        create_folder([save_folder, num2str(snrValue), '\noise_data_white_noise\', ]);
        audiowrite([save_folder, num2str(snrValue), '\noise_data_white_noise\', speciesList(iSpecies).name], noisedSignal_white, fs);
        
        % add pink noise
        if length(cleanSignal) > length(wavin3_pink)
            n_times = 1 + round( length(cleanSignal) / length(wavin3_pink));
            wavin4_pink = repmat(wavin3_pink,n_times, 1);
            [noisedSignal_pink,~] = addnoise(cleanSignal, wavin4_pink(1:length(cleanSignal)), snrValue);
        else
            [noisedSignal_pink,~] = addnoise(cleanSignal, wavin3_pink(1:length(cleanSignal)), snrValue);
        end
        
        create_folder([save_folder, num2str(snrValue), '\noise_data_pink_noise\']);
        audiowrite([save_folder, num2str(snrValue), '\noise_data_pink_noise\', speciesList(iSpecies).name], noisedSignal_pink, fs);
        
        % % add babble noise
        % if length(cleanSignal) > length(wavin3_babble)
        %     n_times = 1 + round(length(cleanSignal)  / length(wavin3_babble) );
        %     wavin4_babble = repmat(wavin3_babble,n_times, 1);
        %     [noisedSignal_babble,~] = addnoise(cleanSignal, wavin4_babble(1:length(cleanSignal)), snrValue);
        % else
        %     [noisedSignal_babble,~] = addnoise(cleanSignal, wavin3_babble(1:length(cleanSignal)), snrValue);
        % 
        % end
        % create_folder([save_folder, num2str(snrValue), '\noise_data_babble_noise\']);
        % audiowrite([save_folder, num2str(snrValue), '\noise_data_babble_noise\', speciesList(iSpecies).name], noisedSignal_babble, fs);
        
        % add rain
        if length(cleanSignal) > length(wavin3_rain)
            n_times = 1 + round( length(cleanSignal) / length(wavin3_rain));
            wavin4_rain = repmat(wavin3_rain,n_times, 1);
            [noisedSignal_rain,~] = addnoise(cleanSignal, wavin4_rain(1:length(cleanSignal)), snrValue);
        else
            [noisedSignal_rain,~] = addnoise(cleanSignal, wavin3_rain(1:length(cleanSignal)), snrValue);
        end
        create_folder([save_folder, num2str(snrValue), '\noise_data_rain_noise\']);
        audiowrite([save_folder, num2str(snrValue), '\noise_data_rain_noise\', speciesList(iSpecies).name], noisedSignal_rain, fs);
        
        % add wind
        if length(cleanSignal) > length(wavin3_wind)
            n_times = 1 + round(length(cleanSignal) / length(wavin3_wind) );
            wavin4_wind = repmat(wavin3_wind,n_times, 1);
            [noisedSignal_wind,~] = addnoise(cleanSignal, wavin4_wind(1:length(cleanSignal)), snrValue);
        else
            [noisedSignal_wind,~] = addnoise(cleanSignal, wavin3_wind(1:length(cleanSignal)), snrValue);
        end
        
        create_folder([save_folder, num2str(snrValue), '\noise_data_wind_noise\']);
        audiowrite([save_folder, num2str(snrValue), '\noise_data_wind_noise\', speciesList(iSpecies).name], noisedSignal_wind, fs);
        
    end
end
%[EOF]



