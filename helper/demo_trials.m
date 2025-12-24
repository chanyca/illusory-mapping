%% Demo trials
% Visual flash detection - show ONLY flash
% Double flash - show flash AND beep

HideCursor;
showTrace = true;

%% Draw fixation
while ~KbCheck
    drawStim("fixation", window, stim, env);
    Screen('Flip', window);   
end
%% First show flash

WaitSecs(0.2); % so it doesn't return true immediately
while ~KbCheck
    line1 = 'For the following experiment,\n';
    line2 = 'you will see flash(es) like this.';
    drawStim([line1 line2], window, stim, env);
    Screen('Flip', window);    
end

WaitSecs(0.2);
while ~KbCheck
    drawStim("flash", window, stim, env);
    Screen('Flip', window);    
end
%% Second show beep

if ismember(task, ["df", "ad"])
    drawStim('You will also hear beep(s) like this.', window, stim, env)
    Screen('Flip', window);
    WaitSecs(1);
    KbWait;
    
    PsychPortAudio('FillBuffer', env.audio_handle, [param.single_beep_array; param.single_beep_array]);
    [~, StimulusOnsetTimePre, FlipTimestamp, ~, ~] = Screen('Flip', window);

    t0 = StimulusOnsetTimePre + env.ifi;
    WaitSecs(1);

    audiostarttime = PsychPortAudio('Start', env.audio_handle, param.repetitions, t0, param.waitForDeviceStart);
    WaitSecs(1);
end

%% Third play beep that indicates question prompt

drawStim('The following tone indicates that it is time to respond', window, stim, env);
Screen('Flip', window);
KbWait;
WaitSecs(1);
Screen('Flip', window);
playAudio('resp', env);
WaitSecs(1);
