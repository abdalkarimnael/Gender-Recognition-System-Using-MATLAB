%%%Name: Abdalkarim Eiss
%%%ID: 1200015
%%%%Gender Recognition System
%%Training Definition
%%%Define the training male file directory 
% Directory containing the .wav files
male_training_dir = 'C:\Users\Asus\Desktop\DSP_ass\Train\MaleF';
% List of male training .wav files in the directory
male_training_files = dir(fullfile(male_training_dir, '*.wav'));
%%%Define the training female .wav directory files
female_training_dir = 'C:\Users\Asus\Desktop\DSP_ass\Train\FemaleF';
% List of male training .wav files in the directory
female_training_files = dir(fullfile(female_training_dir, '*.wav'));
%%%%Testing definitions
%%%Define the testing male .wav directory files
male_testing_dir = 'C:\Users\Asus\Desktop\DSP_ass\Test\MaleF';
% List of male testing .wav files in the directory
male_testing_files = dir(fullfile(male_testing_dir, '*.wav'));

%%%Define the testing female .wav directory files
female_testing_dir = 'C:\Users\Asus\Desktop\DSP_ass\Test\FemaleF';
% List of male testing .wav files in the directory
female_testing_files = dir(fullfile(female_testing_dir, '*.wav'));

%%% ----------Training ------------
male_data = [];  % to store the features of male audio training files
female_data = [];  % to store the features of female audio training files

% Loop through each file for male training data
for i = 1:numel(male_training_files)
    % Load the audio file
    filename = fullfile(male_training_dir, male_training_files(i).name);
    [y, fs] = audioread(filename);

    % Divide the signal into 3 parts and calculate features for each part
    ZCR_m1 = mean(abs(diff(sign(y(1:floor(end/3))))))./2;
    ZCR_m2 = mean(abs(diff(sign(y(floor(end/3):floor (end*2/3))))))./2;
    ZCR_m3 = mean(abs(diff(sign(y(floor(end*2/3):end)))))./2;
    energy_male = sum(y.^2); % Calculate energy
    % Power spectral density
    [psd, ~] = pwelch(y, [], [], [], fs);
    psd_male = mean(psd);

    % Combine features
    features_male = [ZCR_m1 ZCR_m2 ZCR_m3 energy_male psd_male];
    male_data = vertcat(male_data, features_male(:));
end

% Calculate the mean of features for male training data
features_mean_male = mean(male_data);
fprintf('The features mean for Male Audios is \n');
disp(features_mean_male);

% Repeat the process for female training data
for i = 1:numel(female_training_files)
    % Load the audio file
    fName = fullfile(female_training_dir, female_training_files(i).name);
    [y, fs] = audioread(fName);

    % Divide the signal into 3 parts and calculate features for each part
    ZCR_f1 = mean(abs(diff(sign(y(1:floor(end/3))))))./2;
    ZCR_f2 = mean(abs(diff(sign(y(floor(end/3):floor (end*2/3))))))./2;
    ZCR_f3 = mean(abs(diff(sign(y(floor(end*2/3):end)))))./2;
    energy_female = sum(y.^2); % Calculate energy
    % Power spectral density
    [psd_f, ~] = pwelch(y, [], [], [], fs);
    psd_female = mean(psd_f);

    % Combine features
    features_female = [ZCR_f1 ZCR_f2 ZCR_f3 energy_female psd_female];
    female_data = vertcat(female_data, features_female(:));
end

% Calculate the mean of features for female training data
features_mean_female = mean(female_data);
fprintf('The features mean for Female Audios is \n');
disp(features_mean_female);

%%%%%-------------Testing----------------
sum_m = 0;  % To count the successful male classifications
sum_f = 0;  % To count the successful female classifications

fprintf('MALE Testing Results:\n');
% MALE TESTING
for i = 1:numel(male_testing_files)
    % Load the audio file
    fName = fullfile(male_testing_dir, male_testing_files(i).name);
    [y, fs] = audioread(fName);

    % Divide the signal into 3 parts and calculate features for each part
    ZCR_ma1 = mean(abs(diff(sign(y(1:floor(end/3))))))./2;
    ZCR_ma2 = mean(abs(diff(sign(y(floor(end/3):floor (end*2/3))))))./2;
    ZCR_ma3 = mean(abs(diff(sign(y(floor(end*2/3):end)))))./2;
    energy = sum(y.^2); % Calculate energy
    % Power spectral density
    [psd_y, ~] = pwelch(y, [], [], [], fs);
    psd_test = mean(psd_y);

    % Combine features
    features_test = [ZCR_ma1 ZCR_ma2 ZCR_ma3 energy psd_test];

    % Calculate cosine distances
    cosine_dist_male = pdist2(features_test', features_mean_male', 'cosine');
    cosine_dist_female = pdist2(features_test', features_mean_female', 'cosine');

    % Make the decision based on cosine distance
    if (cosine_dist_male > cosine_dist_female)
        fprintf('Test file [Male] #%d classified as FEMALE\n', i);
    else
        fprintf('Test file [Male] #%d classified as MALE\n', i);
        sum_m = sum_m + 1;  % Calculate the sum of successful male files
    end
end

%%% FEMALE TESTING
fprintf('FEMALE Testing Results:\n');
for i = 1:numel(female_testing_files)
    % Load the audio file
    fName = fullfile(female_testing_dir, female_testing_files(i).name);
    [y, fs] = audioread(fName);

    % Divide the signal into 3 parts and calculate features for each part
    ZCR_fe1 = mean(abs(diff(sign(y(1:floor(end/3))))))./2;
    ZCR_fe2 = mean(abs(diff(sign(y(floor(end/3):floor (end*2/3))))))./2;
    ZCR_fe3 = mean(abs(diff(sign(y(floor(end*2/3):end)))))./2;
    energy2 = sum(y.^2); % Calculate energy
    % Power spectral density
    [psd_y2, ~] = pwelch(y, [], [], [], fs);
    psd_test2 = mean(psd_y2);

    % Combine features
    features_test2 = [ZCR_fe1 ZCR_fe2 ZCR_fe3 energy2 psd_test2];

    % Calculate cosine distances
    cosine_dist_male2 = pdist2(features_test2', features_mean_male', 'cosine');
    cosine_dist_female2 = pdist2(features_test2', features_mean_female', 'cosine');

    % Make the decision based on cosine distance
    if (cosine_dist_male2 < cosine_dist_female2)
        fprintf('Test file [Female] #%d classified as MALE\n', i);
    else
        fprintf('Test file [Female] #%d classified as FEMALE\n', i);
        sum_f = sum_f + 1; % Calculate the sum of successful female files
    end
end

%%%%Display and calculate the accuracy for each classification process:
accuracy_m = (sum_m / length(male_testing_files)) * 100;  %%%To calculate the male classification accuracy
fprintf('Accuracy of classification for the male files is %.2f', accuracy_m);
disp('%');
%%%Acuuracy calculations for the female classification
accuracy_f = (sum_f / length(female_testing_files)) * 100;  %%%To calculate the female classification accuracy
fprintf('Accuracy of classification for the female files is %.2f', accuracy_f);
disp('%');

