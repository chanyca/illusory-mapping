function drawStim(target, window, stim, env)

if string(target) == "fixation"
    % Screen('DrawLines', window, stim.fix_coords, stim.fix_lw, env.white, [env.xCenter env.yCenter], 2);
    Screen('DrawLines', window, stim.fix_coords, stim.fix_lw, env.dark_grey, [env.xCenter env.yCenter], 2);
elseif string(target) == "flash"
    Screen('FillOval', window, 1, CenterRectOnPointd(stim.baseCircle, env.xCenter, env.yCenter));
else % text prompt
    % DrawFormattedText(window, target, 'center', 'center', env.white);
    DrawFormattedText(window, target, 'center', 'center', env.dark_grey);
end