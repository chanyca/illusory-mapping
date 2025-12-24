function [responded, endTask, secs, num] = checkKey  
    num = 99;
    clear keyCode
    KbName('UnifyKeyNames')
    responded = false;
    endTask = false;
    
    [keyIsDown, secs, keyCode] = KbCheck;

    if keyCode(KbName('ESCAPE'))
        responded = true;
        endTask = true;
        num = 99;
    elseif keyCode(KbName('1')) || keyCode(KbName('1!')) || keyCode(35)
        num = 1;
    elseif keyCode(KbName('2')) || keyCode(KbName('2@')) || keyCode(40)
        num = 2;
    elseif keyCode(KbName('3')) || keyCode(KbName('3#')) || keyCode(34)
        num = 3;
    elseif keyCode(KbName('0')) || keyCode(KbName('0)')) || keyCode(45)
        num = 0;
    elseif keyCode(KbName('LeftArrow')) || keyCode(28)
        num = 1;
    elseif keyCode(KbName('RightArrow')) || keyCode(29)
        num = 0;
    elseif keyCode(KbName('Return'))
        num = 1;
    elseif IsWin && keyCode(KbName('BackSpace'))
        num = 7;
    elseif IsOSX && keyCode(KbName('DELETE'))
        num = 7;
    end

    while KbCheck; end % Wait until all keys are released

    if num <= 4
        responded = true;        
    end
 
end