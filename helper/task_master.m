%% Master script for running trials


%% Extract conditions
loc_list     = Data.Conditions(:,1);
beeps_list   = Data.Conditions(:,2);
flashes_list = Data.Conditions(:,3);

%% Check task type
if task=="vf" % Visual Flash Detection Task   
    playBeep = false;
elseif ismember(task, ["df", "ad"]) % Double Flash Task, Beep Detection Task
    playBeep = true;    
end

%% Wait for key press to begin experiment
line1 = 'The experiment will begin as soon as you are ready\n\n';
line2 = 'Press any key to continue.';
drawStim([line1 line2], window, stim, env);
Screen('Flip', window);
KbWait

%% start showing stimuli

WaitSecs(1);

% count no of nonempty trials
startTrial = length(Data.Responses) + 1;
% startTrial = size(Data.Conditions,1)-2;
for n=startTrial:size(Data.Conditions,1)

    responded = false;
    while ~responded
        if mod(n, 20) == 0 && n ~= 0 % break every 20 trials
            % intermission
            line1 = 'Please take a break if needed\n';
            line2 = '\n Press any key to continue';            
            drawStim([line1 line2], window, stim, env);
            % playAudio('break', env);

            % get progress
            drawStatusBar(n, size(Data.Conditions,1), env, window);
            Screen('Flip', window); 
            
            [keyIsDown, secs, keyCode] = KbCheck;
            if keyCode(KbName('ESCAPE'))
                cleanup;
            end
            KbStrokeWait;
        end
    
        Xposition = circLoc(loc_list(n)).xpos;
        Yposition = circLoc(loc_list(n)).ypos;
    
        num_flashes = flashes_list(n);
        num_beeps = beeps_list(n);

        % ===== for timing test !!!! COMMENT OUT FOR EXPERIMENT =========
%         Xposition = circLoc(Data.Conditions(1,1)).xpos;
%         Yposition = circLoc(Data.Conditions(1,1)).ypos;%     
%         num_flashes = 1;
%         num_beeps = 2;
        % ================================================================
    
        dispStim;
    
        t0 = GetSecs; % get RT
        get_input;
    end
end

%% end

line1 = 'You are now finished with this run.\n';
line2 = '\n Please ring the bell and notify the experimenter. ';
line3 = '\n\n Thank you for your time! ';
line4 = '';

drawStim([line1 line2 line3 line4], window, stim, env);
Screen('Flip', window);
playAudio('outro', env);
KbStrokeWait;


