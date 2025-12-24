function [trials] = genTrials_vf
% Conditions:
% F0B0 - 1 rep per location
% F1B0 - 5 rep per location

nLoc = 24;

%% F1B0
loc_f1b0   = repelem((1:nLoc)', 5);      % 5 reps per location
flash_f1b0 = ones(numel(loc_f1b0), 1);   % 1 flash
beep_f1b0  = zeros(numel(loc_f1b0), 1);  % 0 beep

%% F0B0
loc_f0b0   = repelem((1:nLoc)', 1);        % 1 rep per location
flash_f0b0 = zeros(numel(loc_f0b0), 1);    % 0 flash
beep_f0b0  = zeros(numel(loc_f0b0), 1);    % 0 beep 

%% Combine trials
location = [loc_f1b0;   loc_f0b0];
n_flash  = [flash_f1b0; flash_f0b0];
n_beep   = [beep_f1b0;  beep_f0b0];

%% shuffle
idx = randperm(numel(location));
trials = [location(idx), n_beep(idx), n_flash(idx)];
