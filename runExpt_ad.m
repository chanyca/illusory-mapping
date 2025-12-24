%% Main script to run experiment
%
% 3 - Beep Detection Task
% Conditions:
% F0B0, F0B1, F0B2, F1B0, F1B1, F1B2
%
%% prep
close all;
clear all;
sca;
tic;

% add `helper` to path
addpath(genpath('helper'))
addpath(genpath('dependencies'))
d = Directory;
task = 'ad';

%% Ask for subject details

[Answer,Cancelled] = getInfo('eye', false, 'glasses', false);
if Cancelled
    return
end

showDemo = Answer.demo-1;

date = datetime('now', 'Format','yyyy_MM_dd');
fullFileName = fullfile(d.dataDir, sprintf('%s_%s_%s', ...
    [Answer.group Answer.sid], date, task));

%% returns if subject completed task
pattern = fullfile(d.dataDir, sprintf('%s*%s.mat', ...
    [Answer.group Answer.sid], task));

fileList = dir(pattern);

if ~isempty(fileList) % load file if not empty
    load(fullfile(d.dataDir, fileList.name));
    if Data.complete
        clc
        disp("ERROR: Subject completed this already.")
        sca
        return
    else % Data exists but is incomplete
        fullFileName = fullfile(d.dataDir, extractBefore(fileList.name, '.mat'));
    end
else % Data does not exist
    % initialize data structure
    Data.SubjectID      = [Answer.group Answer.sid];
    Data.complete       = 0;
    Data.Demographic    = [num2str(Answer.age) Answer.sex];
    Data.Conditions     = [];
    Data.Responses      = [];
    Data.RT             = [];

    % generate trials
    [trials] = genTrials_ad;
    Data.Conditions = trials;    
end

clear fileList

%% initial parameters
params;

%% Start experiment
init_sound;
     
% for temporal accuracy
topPriorityLevel = MaxPriority(window);

% show demo
if showDemo
    demo_trials;
end

task_master;

%% Save data

Data.complete = 1;
save([fullFileName, '.mat'], 'Data');
cleanup;