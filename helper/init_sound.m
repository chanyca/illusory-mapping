%% init_sound


PsychPortAudio('Stop', env.audio_handle, 1);
try
    PsychPortAudio('Stop', pahandle1, 1);
    Snd('Close', 1);
end


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
