function [trials] = genTrials_df
% Conditions:
% F1B1 - 1 rep per location
% F1B2 - 10 rep per location

nLoc = 24;

%% F1B2
loc_f1b2   = repelem((1:nLoc)', 10);      % 5 reps per location
flash_f1b2 = ones(numel(loc_f1b2), 1);    % 1 flash
beep_f1b2  = ones(numel(loc_f1b2), 1)*2;  % 2 beep

%% F1B1
loc_f1b1   = repelem((1:nLoc)', 1);       % 1 rep per location
flash_f1b1 = ones(numel(loc_f1b1), 1);    % 1 flash
beep_f1b1  = ones(numel(loc_f1b1), 1);    % 1 beep 

%% Combine trials
location = [loc_f1b2;   loc_f1b1];
n_flash  = [flash_f1b2; flash_f1b1];
n_beep   = [beep_f1b2;  beep_f1b1];

%% shuffle
idx = randperm(numel(location));
trials = [location(idx), n_beep(idx), n_flash(idx)];
