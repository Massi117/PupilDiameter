% This scriot uses data from open source pupilometry data
clear;

data = readtable('run1.csv');
%%
times = data.timecode - data.timecode(1);
area = data.pupil_area;
diameters = 2*sqrt(area./pi);
avgD = mean(diameters);
normD = 100.*(diameters./avgD);

% Plot
figure(2);
plot(times,normD)
title("Figure 4: Pupil Diameter Predictions of mEye");
xlabel('Time (s)');
ylabel("% Signal");
xlim([1 30]);