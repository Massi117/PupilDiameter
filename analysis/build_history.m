% This script analyzes and plots the eye data
clear;

% Load the data
load('data/data_dict.mat');     % loads variable "diameters" & "frame_count"
%stimonset = load('data/run1_timepoints.mat').stimonset;

% Convert stim onsets to frames
%stimonset(stimonset==0) = [];
%onsets = stimonset-stimonset(1);
%onsets = int64(onsets*30);


%% Plot Data
figure(1);
hold on;
plot(frames, diameters);

%for i = 1:length(onsets)
    %xline(onsets(i));
%end
title("Pupil Diameter Changes");
xlabel('Frame');
ylabel("Pixels");
