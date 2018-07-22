
PoptdBm = 10*log10(11);    % the pump laser power, in dBm
source = linspace(-80,20,10);      % the input microwave power, in dBm
%source = -80

a = source*0;   % the convertd optical photon number
aout = source*0;    
Pout = 0*source;    % the output converted optical power
Popt =11e-3;

p = getparams(2e-32*sqrt(1/3),2e-32*sqrt(2/3),1/20e-6,1/50e-6); % sets the parameters of the erbium actoms

%% calculates theconverte signa for each input mw power.
for mm=1:1:length(source)
    if mm ==1;
       [a(mm),aout(mm),Pout(mm)] = cal_aout(Popt,source(mm),1e-5,p);
    else
    [a(mm),aout(mm),Pout(mm)] = cal_aout(Popt,source(mm),a(mm-1),p);
    end
    mm
end

%% plots the data
figure(13);hold on;plot(source(1:end),10*log10(abs(Pout(1:end))),'+');
figure(14);hold on;plot(source(1:end),(abs(Pout(1:end))./(10.^(source(1:end)/10))),'+');