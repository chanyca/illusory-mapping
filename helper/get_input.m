%% get_input

WaitSecs(0.01);
responded = false;
clear keyCode

t0 = GetSecs;

drawPrompt(task, window, stim, env);
playAudio('resp', env);

while ~responded
    drawPrompt(task, window, stim, env);
    
    [responded, endTask, secs, num] = checkKey;
    if endTask
        cleanup;
    end
end

% Save response and RT
Data.Responses(n,:) = num;
Data.RT(n,:) = secs-t0;

%% helper function(s)
function drawPrompt(task, window, stim, env)
    if task=="vf" % Visual Flash Detection Task   
        prompt = 'Did you see a flash?\n\n <== Yes        No ==>';
        ypos = 'center';
    elseif task=="df"% Double Flash Task
        drawStim('FLASH(ES)', window, stim, env);
        prompt = 'How many?\n\n0    1    2    3';
        ypos = env.yCenter-400;
    elseif task=="ad" % Beep Detection Task
        drawStim('BEEP(S)', window, stim, env);
        prompt = 'How many?\n\n0    1    2    3';
        ypos = env.yCenter-400;
    end

    % DrawFormattedText(window, prompt, 'center', ypos, env.white);
    DrawFormattedText(window, prompt, 'center', ypos, env.dark_grey);
    Screen('Flip', window);
end
