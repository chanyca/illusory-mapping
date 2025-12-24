%% cleanup

PsychPortAudio('Stop', env.audio_handle, 1);
try
    PsychPortAudio('Stop', pahandle1, 1);
    Snd('Close', 1);
end

%% save data
save([fullFileName, '.mat'], 'Data');
disp(['Save Complete, fileName is ', [fullFileName, '.mat']])

%% end everything
close all;
sca;