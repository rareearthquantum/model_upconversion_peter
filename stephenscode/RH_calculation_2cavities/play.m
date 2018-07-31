% c = 10;
% mygrid = @(x,y) ndgrid((-x:x/c:x),(-y:y/c:y))*2;
% [x] = mygrid(pi,2*pi);
% 
% z = sin(x);
% % figure(11);mesh(x,y,z)
% 
% % [a1,a2] = arrayfun(@(x,y) mygrid(x,y), [1],[4])
load('0MHz_cav_detuning_821points.mat');
figure(13);hold on;plot(source(1:end),10*log10(abs(Pout(1:end))),'+');
figure(14);hold on;plot(source(1:end),(abs(Pout(1:end))./(10.^(source(1:end)/10)*1e-3)/20e3),'+');