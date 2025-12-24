% convert MP3 to WAV

clear; clc;
files = dir( '*.mp3');
fileList = {files.name};
fileList = strrep(fileList, '.mp3', '');

% check if WAV file exists
for i=1:length(fileList)
    if ~exist([fileList{i} '.wav'], 'file')
        [signal, Fs] = audioread([fileList{i} '.mp3']);
        audiowrite([fileList{i} '.wav'], signal, Fs);
    end

end