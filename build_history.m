% This script analyzes and plots the eye data
clear;

% Load the data
load('data\diameter.mat');
load('data\frames.mat');
load('data\AVPVT108b-run2-redo_audvisPVT_dotPVT.mat')

%d = d(1:2:24276);
%f = f(1:2:24276);

% Plot
figure(1);
plot(f,d);
title("Figure 3: Pupil Diameter Changes");
xlabel('Time (s)');
ylabel("% Signal");

%%

zero_idx = find(d == 0);
zero_frames = f(zero_idx);

nonzero_idx = find(d ~= 0);
nonzero_frames = f(nonzero_idx);
d_nonzeros = nonzeros(d);
baseline = mean(d_nonzeros);

d = 100*double(d)/baseline;

indices = find(abs(d)>200);
d(indices) = [];
f(indices) = [];
idx = double(f)/30;

% Plot
figure(1);
plot(idx,d);
title("Figure 3: Pupil Diameter Changes");
xlabel('Time (s)');
ylabel("% Signal");
xlim([1 30]);
