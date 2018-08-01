addpath('C:\Users\labrat\Desktop\code\qotoolbox\qotoolbox');

   
 source = linspace(-70,30,101);      % microwave power, in dBm
%  source = -9;      % microwave power, in dBm

a = source*0;b = a;
generation=a;
absorption=a;
aout = source*0;
Pout = 0*source;
Popt = 11e-3;    % pump laser power, in W

p = getparams(2e-32*sqrt(2/3),2e-32*sqrt(1/3),1/1e-6,1/1e-6);
%p = getparams(d31,d32,gamma3d,gamma2d);

for mm=1:1:length(source)
    if mm ==1;
       [a(mm),aout(mm),Pout(mm),b(mm),generation(mm),absorption(mm)] = cal_aout(Popt,source(mm),1e-5,1e-5,p);
    else
    [a(mm),aout(mm),Pout(mm),b(mm),generation(mm),absorption(mm)] = cal_aout(Popt,source(mm),a(mm-1),b(mm-1),p);
    end
    mm
end

figure(13);hold on;plot(source(1:end),10*log10(abs(Pout(1:end))),'o');
figure(14);hold on;plot(source(1:end),((abs(Pout(1:end)/1e-3)./(10.^(source(1:end)/10)))/2e4),'o');
% save('temp.mat');
save('0MHz_cav_detuning_4021points_11mw_wopt6mm_-70to30.mat');