%% Calculate VA
% Screen width:          60 cm = 600 mm
% Horizontal resolution: 3840
% pixel size:            600/3840 = 0.1562
% Viewing distance:      57 cm = 570 mm
% VA formula:            2 * atand ( size in mm / viewDist*2)

param.pixSize = 600/env.screenXpixels;
param.viewDist = 570*2;

deg.p25 = (tand(.25)/2*param.viewDist)/param.pixSize;
deg.half = (tand(.5)/2*param.viewDist)/param.pixSize;
deg.one = (tand(1)/2*param.viewDist)/param.pixSize;
deg.two = (tand(2)/2*param.viewDist)/param.pixSize;
deg.three = (tand(3)/2*param.viewDist)/param.pixSize;
deg.four = (tand(4)/2*param.viewDist)/param.pixSize;
deg.five = (tand(5)/2*param.viewDist)/param.pixSize;
deg.ten = (tand(10)/2*param.viewDist)/param.pixSize;
deg.fifteen = (tand(15)/2*param.viewDist)/param.pixSize;

deg.five_diag = sqrt(power(deg.five,2)/2);
deg.ten_diag = sqrt(power(deg.ten,2)/2);
deg.fifteen_diag = sqrt(power(deg.fifteen,2)/2);