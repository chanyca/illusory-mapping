function playAudio(target, env)

addpath('audio\')

if string(target) == "resp"
    % play sound to indicate time to respond
    snddata = MakeBeep(300, 0.5, env.sampleRate);
    PsychPortAudio('FillBuffer', env.audio_handle, [snddata; snddata]);
    PsychPortAudio('Volume', env.audio_handle, 0.1);
    t1 = PsychPortAudio('Start', env.audio_handle, 1, 0, 1);
    PsychPortAudio('Stop', env.audio_handle, 3);

elseif string(target) == "warning"
    % play warning audio file
    [y, freq] = psychwavread('audio\warning.wav');
    wavedata = y';
    nrchannels = size(wavedata,1);
    pahandle1 = PsychPortAudio('Open', [], [], 0, freq, nrchannels);
    PsychPortAudio('FillBuffer', pahandle1, wavedata);
    PsychPortAudio('Volume', pahandle1, 0.3);
    PsychPortAudio('Start', pahandle1, 1, 0, 1);
    PsychPortAudio('Stop', pahandle1, 3);

elseif string(target) == "break"
    % play break audio file
    [y, freq] = psychwavread('audio\break.wav');
    wavedata = y';
    nrchannels = size(wavedata,1);
    pahandle1 = PsychPortAudio('Open', [], [], 0, freq, nrchannels);
    PsychPortAudio('FillBuffer', pahandle1, wavedata);
    PsychPortAudio('Volume', pahandle1, 0.3);
    PsychPortAudio('Start', pahandle1, 1, 0, 1);
    PsychPortAudio('Stop', pahandle1, 3);

elseif string(target) == "outro"
    % play congrats audio file
    [y, freq] = psychwavread('audio\congrats.wav');
    wavedata = y';
    nrchannels = size(wavedata,1);
    pahandle1 = PsychPortAudio('Open', [], [], 0, freq, nrchannels);
    PsychPortAudio('FillBuffer', pahandle1, wavedata);
    PsychPortAudio('Volume', pahandle1, 0.3);
    PsychPortAudio('Start', pahandle1, 1, 0, 1);
    PsychPortAudio('Stop', pahandle1, 3);
end



end