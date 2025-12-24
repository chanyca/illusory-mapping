function drawStatusBar(n, maxN, env, window)

length = 2000;
width = 250;
padding = (env.screenXpixels - length)/2;

prog = n/maxN;
Screen('FrameRect', window, env.white, [padding 100 env.screenXpixels-padding width]); % left, top, right, bottom
Screen('FillRect', window, env.white, [padding 105 padding+length*prog width-5]);

% Screen('TextSize', window, param.text_size);
DrawFormattedText(window, [num2str(round(prog*100)), '%'], padding+length*prog-50, width+100, env.white);


end