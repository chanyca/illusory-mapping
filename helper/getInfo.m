%% getInfo
function [Answer,Cancelled] = getInfo(varargin)

eye = true; glasses = true; demo = true;
while ~isempty(varargin)
    switch lower(varargin{1})
        case 'eye'
            eye = varargin{2};
        case 'glasses'
            glasses = varargin{2};        
        case 'demo'
            demo = varargin{2};          
    end
    varargin(1:2) = [];
end


Title = '';

Options.Resize = 'on';
Options.Interpreter = 'tex';
Options.CancelButton = 'on';
Options.ApplyButton = 'off';
Options.ButtonNames = {'Continue','Cancel'}; %<- default names, included here just for illustration
Options.AlignControls = 'on';

Prompt = {};
Formats = {};
DefAns = struct([]);

% Group
Prompt(1,:) = {'Group', 'group', []};
Formats(1,1).type = 'list';
Formats(1,1).format = 'text';
Formats(1,1).style = 'radiobutton';
Formats(1,1).items = {'SV' 'LV'};
DefAns(1).group = 'SV';

% Subject ID
Prompt(end+1,:) = {'Subject ID', 'sid', []};
Formats(1,2).type = 'edit';
Formats(1,2).format = 'text';
Formats(1,2).size = 50; % automatically assign the height
DefAns.sid = '999';

% Sex assigned at birth
Prompt(end+1,:) = {'Sex assigned at birth','sex',[]};
Formats(2,1).type = 'list';
Formats(2,1).format = 'text';
Formats(2,1).style = 'radiobutton';
Formats(2,1).items = {'F' 'M'};
DefAns.sex = 'F';

% Age
Prompt(end+1,:) = {'Age', 'age', []};
Formats(2,2).type = 'edit';
Formats(2,2).format = 'integer';
Formats(2,2).size = 50; % automatically assign the height
DefAns.age = 99;

% Eye being tested
if eye
Prompt(end+1,:) = {'Eye being tested ','eye',[]};
Formats(end+1,1).type = 'list';
Formats(end,1).format = 'text';
Formats(end,1).style = 'radiobutton';
Formats(end,1).items = {'L' 'R'};
DefAns.eye = 'L';
end

% Glasses
if glasses
Prompt(end+1,:) = {'Glasses','glasses',[]};
Formats(end+1,1).type = 'list';
Formats(end,1).format = 'text';
Formats(end,1).style = 'radiobutton';
Formats(end,1).items = {'with' 'without' 'na'};
DefAns.glasses = 'na';
end

% Demo
if demo
Prompt(end+1,:) = {'Show demo','demo',[]};
Formats(end+1,1).type = 'list';
Formats(end,1).format = 'integer';
Formats(end,1).style = 'radiobutton';
Formats(end,1).items = [0 1];
DefAns.demo = 2; % index, default show demo
end

%% FINAL STEP
[Answer,Cancelled] = inputsdlg(Prompt,Title,Formats,DefAns,Options);

end
