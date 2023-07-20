% This script analyzes and plots the eye data
clear;

% Load the data
load('data\diameter.mat');
load('data\frames.mat');
load('data\AVPVT108b-run2-redo_audvisPVT_dotPVT.mat')

% Smooth the data
n = 10; % average every n values

zero_idx = find(d == 0);
zero_frames = f(zero_idx);

nonzero_idx = find(d ~= 0);
nonzero_frames = f(nonzero_idx);
d_nonzeros = nonzeros(d);
baseline = mean(d_nonzeros);

smooth_d = [];

for i = 1:n:numel(d_nonzeros)
    m = mean(d_nonzeros(i));
    m = ones(1,n)*m;
    smooth_d = [smooth_d m];
end

smooth_d = [smooth_d zeros(1,numel(zero_frames))];
smooth_f = [nonzero_frames zero_frames];

[smooth_f,sortIdx] = sort(smooth_f,'ascend');
smooth_d = smooth_d(sortIdx);

smooth_percent = 100*smooth_d/baseline;

% Plot
figure(1);
plot(smooth_f,smooth_percent);
title("Pupil Diameter Changes");
xlabel('Frame');
ylabel("% Signal");
