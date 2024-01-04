% This script analyzes and plots the eye data
clear;

% Load the data
load('data\diameters.mat');     % loads variable "diameters"
load('data\frame_count.mat');   % loads variable "frame_count"
stimonset = load('data\run1_timepoints.mat').stimonset;

% Convert stim onsets to frames
stimonset(stimonset==0) = [];
onsets = stimonset-stimonset(1);
onsets = int64(onsets*30);


%% Plot Data
figure(1);
hold on;
plot(frame_count, diameters);

for i = 1:length(onsets)
    xline(onsets(i));
end
title("Pupil Diameter Changes");
xlabel('Frame');
ylabel("Pixels");
