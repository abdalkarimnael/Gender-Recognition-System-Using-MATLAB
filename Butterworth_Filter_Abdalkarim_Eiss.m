%%Name: Abdalkarim Eiss --- ID: 1200015
%%%%This code is to filter the .wav voices sice it is recorded using
%%%%mobiles
% Define the sampling frequency
fs = 44100;

% Design a low-pass filter (Butterworth filter)
order = 4; % Order of the filter
cutoff_frequency = 5000; % Cutoff frequency in Hz
[b, a] = butter(order, cutoff_frequency / (fs/2), 'low');

% Directory containing the .wav files
input_directory = 'C:/Users/Asus/Desktop/DSP_ass/OriginalAudio/Female';

% List of .wav files in the directory
file_list = dir(fullfile(input_directory, '*.wav'));

% Loop through each file and apply the filter
for i = 1:numel(file_list)
    % Load the audio file
    filename = fullfile(input_directory, file_list(i).name);
    [audio, fs_original] = audioread(filename);

    % Ensure the audio is at the desired sampling frequency
    if fs_original ~= fs
        audio = resample(audio, fs, fs_original);
    end

    % Apply the filter
    filtered_audio = filter(b, a, audio);

    % Save the filtered audio to a new file
    output_filename = fullfile('C:/Users/Asus/Desktop/DSP_ass/Train/FemaleF', ['filtered_', file_list(i).name]);
    audiowrite(output_filename, filtered_audio, fs);
end

disp('Filtering complete.');
