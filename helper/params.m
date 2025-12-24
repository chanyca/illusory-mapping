%% prep screen 
PsychDefaultSetup(2);
Screen('Preference', 'SkipSyncTests', 1); 
env.screens = Screen('Screens'); % Get the screen numbers
env.screenNumber = max(env.screens);
env.white = WhiteIndex(env.screenNumber);
env.black = BlackIndex(env.screenNumber);
env.grey = env.white / 2;
env.dark_grey = 0.3;

[window, windowRect] = PsychImaging('OpenWindow', env.screenNumber, env.black); % open window
[env.screenXpixels, env.screenYpixels] = Screen('WindowSize', window); % window size
env.ifi = Screen('GetFlipInterval', window); % frame rate

env.FR = round(1/env.ifi); % 60
[env.xCenter, env.yCenter] = RectCenter(windowRect); % where is center
% Set up alpha-blending for smooth (anti-aliased) lines
Screen('BlendFunction', window, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');

%% prep sound

InitializePsychSound(3);
param.nchannels = 2;
param.startCue = 0;
param.repetitions = 1;
param.waitForDeviceStart = 1; 

env.audio_handle = PsychPortAudio('Open', [], 1, 1, [], param.nchannels, []);
s = PsychPortAudio('GetStatus', env.audio_handle);
env.sampleRate = s.SampleRate;

param.beepDuration = 0.007; %s (7 ms)
param.audioDelay = -0.105;
param.preFlashBeepT = 0.023 - param.audioDelay; % From double flash paper;
param.postFlashBeepT = 0.064 - (0.023 - param.audioDelay); % From double flash paper;

% flash will begin at t = 50ms
param.beepOneT = 0.170 - param.preFlashBeepT; % in sec
param.beepTwoT = 0.170 + param.postFlashBeepT; % in sec
param.beepOneSamps = int64(param.beepOneT * env.sampleRate); % in Hz
param.beepTwoSamps = int64(param.beepTwoT * env.sampleRate); % in Hz
param.beep_stim = MakeBeep(800, param.beepDuration, env.sampleRate);
[foo, param.beep_samps] = size(param.beep_stim);
disp("param.beep_samps" + param.beep_samps)


% Entire stimulus is 200 ms long
param.beep_nSamps = int64(env.sampleRate * 0.200); % 9600

param.zero_beep_array = zeros(1, param.beep_nSamps);

param.single_beep_array = zeros(1, param.beep_nSamps);
param.single_beep_array(1, param.beepOneSamps:param.beepOneSamps + param.beep_samps - 1) = param.beep_stim * 0.5;

param.double_beep_array = param.single_beep_array;
param.double_beep_array(1, param.beepTwoSamps:param.beepTwoSamps + param.beep_samps - 1) = param.beep_stim * 0.5;


%% keys
KbName('UnifyKeyNames'); 

%% eccentricity
param.eccs = 600;
param.shift = 0;
visual_angle;

param.cross = [deg.five, deg.ten, deg.fifteen];
param.diago = [deg.five_diag, deg.ten_diag, deg.fifteen_diag];

for i=1:24
    switch i
        case num2cell(1:3) % 1-6 actually - Vertical axis
            circLoc(i).xpos = env.xCenter;
            circLoc(i).ypos =  env.yCenter + param.cross(i);
            circLoc(i+3).xpos = env.xCenter;
            circLoc(i+3).ypos = env.yCenter - param.cross(i);
        case num2cell(7:9) % 7-12 actually - Horizontal axis
            circLoc(i).xpos = env.xCenter + param.cross(i-6);
            circLoc(i+3).xpos =  env.xCenter - param.cross(i-6);
            circLoc(i).ypos = env.yCenter;
            circLoc(i+3).ypos = env.yCenter;
        case num2cell(13:15) % 13-24
            % (+,+) bottom right
            circLoc(i).xpos = env.xCenter + param.diago(i-12);
            circLoc(i).ypos = env.yCenter + param.diago(i-12);
            % (-,-) top left
            circLoc(i+3).xpos = env.xCenter - param.diago(i-12);
            circLoc(i+3).ypos = env.yCenter - param.diago(i-12);
            % (+,-) top right
            circLoc(i+6).xpos = env.xCenter + param.diago(i-12);
            circLoc(i+6).ypos = env.yCenter - param.diago(i-12);
            % (-,+) bottom left
            circLoc(i+9).xpos = env.xCenter - param.diago(i-12);
            circLoc(i+9).ypos = env.yCenter + param.diago(i-12);
    end
end

%% prep visual stimuli
% param.distX = 100;
stim.duration = .2; %200 ms
stim.nFrames = round(stim.duration / env.ifi);
stim.baseCircle = [0 0 deg.two deg.two];

% fixation
stim.fix_size = 50;
stim.fix_lw = 10;
stim.fix_X = [-stim.fix_size  stim.fix_size  0               0];
stim.fix_Y = [0               0              -stim.fix_size  stim.fix_size];
stim.fix_coords = [stim.fix_X; stim.fix_Y];
stim.fix_dur = .5; %1; % in sec
stim.fix_nFrames = round(stim.fix_dur / env.ifi);
stim.fix_deg_allowed = deg.two;
stim.fix_circle = [env.xCenter - stim.fix_deg_allowed; ...
                   env.yCenter- stim.fix_deg_allowed; ...
                   env.xCenter + stim.fix_deg_allowed; ...
                   env.yCenter + stim.fix_deg_allowed];
stim.fix_circle_width = 30;

%% text
param.text_color = 0.4; %env.white; %mod(backgr + 0.5,1);
param.text_size = 100;
Screen('TextSize', window, param.text_size);
Screen('TextStyle', window, 1);





