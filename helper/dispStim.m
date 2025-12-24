%% dispStim.m

HideCursor;

%% audio buffer
% entire stimulus is 200ms long
% prep beep here
disp(num_beeps)
if playBeep    
    % Fill buffer
    if num_beeps == 0
        PsychPortAudio('FillBuffer', env.audio_handle, [param.zero_beep_array; param.zero_beep_array]);
    elseif num_beeps == 1
        PsychPortAudio('FillBuffer', env.audio_handle, [param.single_beep_array; param.single_beep_array]);
    elseif num_beeps == 2
        PsychPortAudio('FillBuffer', env.audio_handle, [param.double_beep_array; param.double_beep_array]);
    end
end

%% fixation
for iframe = 1:stim.fix_nFrames
    drawStim("fixation", window, stim, env);
    Screen('Flip', window);
end

%% actual stim

% ________ for temporal accuracy check _________
% num_flashes = 2;
% Xposition = circLoc(1).xpos;
% Yposition = circLoc(1).ypos;
% ______________________________________________

% Pre-flip, t approx -0.0167
drawStim("fixation", window, stim, env)
[~, StimOnsetTimePre, FlipTimestamp, ~, ~] = Screen('Flip', window);

t0 = StimOnsetTimePre + env.ifi;

if playBeep
    PsychPortAudio('Volume', env.audio_handle, 1);
    audiostarttime = PsychPortAudio('Start', env.audio_handle, param.repetitions, t0, param.waitForDeviceStart);
end

if num_flashes >= 1    
    for iframe = 1:stim.nFrames % 200 ms / 12 frames   
        if iframe == 4
            drawStim("fixation", window, stim, env);
            Screen('FillOval', window, env.white, CenterRectOnPointd(stim.baseCircle, Xposition, Yposition));
            Screen('Flip', window);
        else
            % fixation for 3 frames before flash
            drawStim("fixation", window, stim, env);    
            Screen('Flip', window);
        end
    end

else % no flash, keep drawing fixation
    for iframe = 1:stim.nFrames
        drawStim("fixation", window, stim, env);
        Screen('Flip', window);
    end
end


%% fixation
for iframe = 1:stim.fix_nFrames
    drawStim("fixation", window, stim, env);
    Screen('Flip', window);
end
