function [trials] = genTrials_ad
% Conditions:
% 3 reps per location
% F0B0, F0B1, F0B2
% F1B0, F1B1, F1B2

nLoc = 24;
nRep = 3;

conds = [
    0 0;  % F0B0
    0 1;  % F0B1
    0 2;  % F0B2
    1 0;  % F1B0
    1 1;  % F1B1
    1 2;  % F1B2
];

%% 

location = [];
n_beep = [];
n_flash = [];

for c = 1:size(conds,1)
    for i = 1:nLoc
        location = [location;   repmat(i, nRep, 1)];
        n_beep   = [n_beep;  repmat(conds(c,2), nRep, 1)];
        n_flash  = [n_flash; repmat(conds(c,1), nRep, 1)];
    end
end

%% shuffle
idx = randperm(numel(location));
trials = [location(idx), n_beep(idx), n_flash(idx)];
