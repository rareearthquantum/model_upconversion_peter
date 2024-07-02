% Fit a couple of curves and determine T1 from our measurements.
% GK, 17-Feb-MMXIX
addpath('~/ltj-CaltechMeasurements/BluFors2-Logs/')

nums=[2,3,4,5,6];
tta_all=zeros(5,20);
ttb_all=zeros(5,20);
f1a_all=zeros(5,20);
f1b_all=zeros(5,20);

tta2_all=zeros(5,20);
ttb2_all=zeros(5,20);
f1a2_all=zeros(5,20);
f1b2_all=zeros(5,20);
fc=4733
%trans_all=zeros(201,60,ii);
for ii_num = 1:5
    load(['09_CALTECH_FF',num2str(nums(ii_num)),'.mat'])

    


filename = char(filename)
notes = char(notes)
% Temperatures: 
timestamps_datenum = timestamp-(max(timestamps)-timestamps)/60/60/24;
temperatures = blufors2_temps(min(min(timestamps_datenum)),...
                              max(max(timestamps_datenum)));
temp_span = [min(temperatures) max(temperatures)];

timestamps_c = timestamps-gradient(timestamps)/2;
timestamps_c = repmat(timestamps_c, length(freq),1);


notes = {sprintf('%f K<T_{MXC}<%f K',temp_span);...
         sprintf('Scriptname: %s',strrep(scriptname,'_','\_'));...
         ' ';...
         'Notes from file:';...
         notes};

notes = textwrap(notes, 36);

% Findpeaks wants positive data; so just shift trans up; move down when done:
%trans_min = min(min(trans));
%trans_min = repmat(trans_min,length(freq),1);
%trans = trans-trans_min;
trans = 10.^(trans/10);% Make a power.


%%
lorentzian = @(params, nu)...
               params(1)*params(2).^2./((nu-params(3)).^2+params(2).^2);
dbllorentz = @(params, nu)...
               lorentzian(params(1:3), nu) + lorentzian (params(4:6),nu);
           
opts = optimset ('Display','none',...
                 'TolFun',1e-8,...
                 'TolX',1e-8);

num_times = length(timestamps);


guesses = nan(num_times, 6);
pp = nan (size(guesses));
ha = [];
for ii = 1:num_times
    %ha(ii) = subplot(6,8,ii);
    %disp(ii)
     

    % Bit iffy hard-coding, but shouldn't be too wrong. I hope.
    [peaks{ii}, loc{ii}] = findpeaks(trans(:,ii), ...
                                    'minpeakheight',2e-7,...
                                    'minpeakdistance',30,...
                                    'npeaks',3,...
                                    'sortstr','descend');
                                %'minpeakprominence',1e-7,...
    % Sort the output of findpeaks; make order more pleasing below: 
    % Can't look for empty peaks(); the cavity low freq is present.
    if ii>N_high_power_spectra && length(loc{ii})>=2
        loc{ii} = loc{ii}(1:2);
        peaks{ii} = peaks{ii}(1:2);
        [loc{ii}, loc_order{ii}] = sort(loc{ii});
        peaks{ii} = peaks{ii}(loc_order{ii});
    end
    
    % Hack around the lack of a big peak by appending a zero to the data
    % for the guesses.
    if length(loc{ii})<2
        if isempty(loc{ii})
           
        else
            loc{ii}(2) = 1;
            peaks{ii}(2) = 0;
        end
    end
    
    %plot(freq(:,ii)',trans(:,ii)',freq(loc{ii},ii),peaks{ii},'x')
    % Now use a lorentzian to fit this:
    guess_heights = peaks{ii};
    guess_centres = freq(loc{ii},ii);
    guess_width = 1;
    
    
    if ii<=N_high_power_spectra
        warning("Skipping this; no relevant peaks to look at...");
    else
        guesses(ii,:) = [guess_heights(1), guess_width, guess_centres(1),...
                         guess_heights(2), guess_width, guess_centres(2)];
        pp(ii,:) = lsqcurvefit(dbllorentz,guesses(ii,:),...
                               freq(:,ii),trans(:,ii), ...
                               [],[],opts);
    end
    
    
%    % plot(freq(:,ii)',trans(:,ii)',...
%          freq(loc{ii},ii),peaks{ii},'x',...
%          freq(:,ii), dbllorentz(guesses(ii,:),freq(:,ii)),...
%          freq(:,ii), dbllorentz(pp(ii,:), freq(:,ii)));
%     
    %grid on
     %legend('measured','peaks','guess','fit')
    % Colour the background on the "pumped" ones:
    %if max(freq(:,ii))<4700
%     if ((ii<=N_high_power_spectra) || ...
%          (ii>(N_low_power_spectra+N_high_power_spectra) &&...
%           ii<=(N_low_power_spectra+2*N_high_power_spectra)))
% 
%         set(ha(ii),'color',[1 1 .2])
%     end
%     % Turn off the axes:
% %     set(get(gca,'xaxis'),'visible','off')
% %     set(get(gca,'yaxis'),'visible','off')
% %     
%     % Label titles for reference:
%     title(sprintf('%u: %.1f s',ii, timestamps(ii)),...
%           'fontsize',5,'fontweight','normal')
%     % Stick to right margin to avoid the y-axis multiplier:
%     set(get(gca,'title'),'horizontalalignment','right')
%     htpos = get(get(gca,'title'),'position');
%     htpos(1) = max(get(gca,'xlim'));
%     htpos(2) = max(max(trans));
%     set(get(gca,'title'),'position',htpos);
%     
%     % Bit of hack, but seems to work:
%     hapos = get(ha(ii),'position');
%     hapos(3) = 0.085;
%     %hapos(4) = 0.095;
%     set(ha(ii),'position',hapos);
%     
end
% set(gcf,'name',filename)
% set(ha, 'ylim', [ min(0,min(min(trans))), max(max(trans))]);
% 
% % Set x and y scales on one image.
% labelled_axes = 41;
% set(get(ha(labelled_axes),'XAxis'),'visible','on')
% set(get(ha(labelled_axes),'YAxis'),'visible','on')
% set(get(ha(labelled_axes),'YAxis'),'ticklabelrotation',0)
% set(get(ha(labelled_axes),'XAxis'),'ticklabelrotation',90)
% set(get(ha(labelled_axes),'XLabel'),'string','f (MHz)')
% set(get(ha(labelled_axes),'YLabel'),'string','S_{21} (lin)')
% set(get(ha(labelled_axes),'Xaxis'),'FontSize',7)
% set(get(ha(labelled_axes),'Yaxis'),'FontSize',7)
% % And the "off-resonance-pump" axes:
% labelled_axes = 1;%N_high_power_spectra;
% set(get(ha(labelled_axes),'XAxis'),'visible','on')
% set(get(ha(labelled_axes),'YAxis'),'visible','on')
% set(get(ha(labelled_axes),'YAxis'),'ticklabelrotation',0)
% set(get(ha(labelled_axes),'XAxis'),'ticklabelrotation',90)
% %set(get(ha(labelled_axes),'XLabel'),'string','f (MHz)')
% set(get(ha(labelled_axes),'YLabel'),'string','S_{21} (lin)')
% set(get(ha(labelled_axes),'Xaxis'),'FontSize',7)
% set(get(ha(labelled_axes),'Yaxis'),'FontSize',7)
% 
% subplot(7,8,[52 56])
% set(get(gca,'XAxis'),'visible','off')
% set(get(gca,'YAxis'),'visible','off')
% set(gca,'color','none')
% hh = text(.08,.7,notes);
% set(hh,'verticalalignment','top')
% 
% hh = text(1,1,strrep(filename,'_','\_'));
% set(hh, 'VerticalAlignment','top','HorizontalAlignment','Right')
% set(hh, 'Fontsize',12, 'Fontweight','Bold')
% 
% 
% 
% set(gcf,'renderer','painters');
% set(gcf,'paperposition',[0.6345    0.6345   19.7310   28.4310]);
% set(gcf,'position',[2 29 761 971])

%%
% 
% figure(9958)
% set(gcf,'position',[2 29 761 971])
% clf
% subplot(4,2,1)
% waterfall(freq',timestamps_c', 10*log10(trans'))
% xlim([4680 4785])
% xlabel('f (MHz)')
% ylabel('t (s)')
% zlabel('S_{21} (dB)')
% xticks = get(gca,'xtick');
% set(gca,'xtick',xticks(2:2:end));
% 
% subplot(4,2,2)
% waterfall(freq',timestamps_c', trans')
% xlim([4680 4785])
% xlabel('f (MHz)')
% ylabel('t (s)')
% zlabel('S_{21} (lin)')
% set(gca,'xtick',xticks(2:2:end));
% 
% 
% subplot(4,2,5)
% plot(timestamps,pp(:,[1 4]),'x',timestamps,guesses(:,[1,4]),'+')
% hl2 = legend('fit','fit','guess','guess','location','best');
% set(hl2, 'fontsize',5)
% xlabel('t(s)')
% ylabel('Amplitude I')
% %title(strrep(filename,'_','\_'))
% set(gcf,'name',filename)

% 
% subplot(4,2,7)
% plot(timestamps, pp(:,[2,5]),'x');%,timestamps,guesses(:,[2,5]),'+')
% hl3 = legend('fit','fit','guess','guess','location','best');
% set(hl3, 'fontsize',5)
% xlabel('t(s)')
% ylabel('\gamma (MHz)')
% set(gcf,'name',filename)


% Now fit the time to the anticrossing frequencies:
% By above mangeling have sorted list of frequencies, which makes things
% helpful.

% Only care about time after the "pump" is turned off; so
start = N_high_power_spectra*2+N_low_power_spectra;
tta = timestamps(start+1:end);
f1a = pp(start+1:end,3);
ttb = timestamps(start+1:end);
f1b = pp(start+1:end,6);

% There's also some relaxing after pumping at the lower frequency:
start2 = N_high_power_spectra+1;
stop2 = N_high_power_spectra+N_low_power_spectra;
tta2 = timestamps(start2:stop2);
f1a2 = pp(start2:stop2,3);
ttb2 = timestamps(start2:stop2);
f1b2 = pp(start2:stop2,6);

opts = optimset ('Display','none',...
                 'TolFun',1e-8,...
                 'TolX',1e-8,...
                 'maxfuneval',1e4,...
                 'maxiter',1e3);

exponential = @(params, t) ...
                params(1).*exp(-(t-params(4))/params(2)) + params(3);
exponential_str = 'A\times e^{\frac{t-t_0}{\tau}} + C';

% Make some guesses (params are A-tau-offset-t_0)
% Know that the "a" will be the lower freq one from sort above, so:
g_a = [max(f1a)-min(f1a),7,min(f1a), min(tta)];
p_a = lsqcurvefit(exponential, g_a, tta, f1a', [],[],opts);

g_b = [min(f1b)-max(f1b),7,max(f1b), min(ttb)];
p_b = lsqcurvefit(exponential, g_b, ttb, f1b', [],[],opts);

g_a2 = [max(f1a2)-min(f1a2),7,min(f1a2), min(tta2)];
p_a2 = lsqcurvefit(exponential, g_a2, tta2, f1a2', [],[],opts);

g_b2 = [min(f1b2)-max(f1b2),7,max(f1b2), min(ttb2)];
p_b2 = lsqcurvefit(exponential, g_b2, ttb2, f1b2', [],[],opts);

% And look at the frequency betweeen the two modes:
start = N_high_power_spectra*2+N_low_power_spectra;
tt = timestamps(start+1:end);
f2a = pp(start+1:end,3);
f2b = pp(start+1:end,6);
f2 = abs(f2a-f2b);

g_2 = [-abs(max(f2)-min(f2)),7, max(f2), min(tt)];
p_2 = lsqcurvefit(exponential, g_2, tt, f2', [], [], opts);

% And the after the lower freq pulse:
tt2 = timestamps(start2:stop2);
f2a2 = pp(start2:stop2,3);
f2b2 = pp(start2:stop2,6);
f22 = abs(f2a2-f2b2);

g_22 = [abs(max(f22)-min(f22)),7, min(f22), min(tt2)];
p_22 = lsqcurvefit(exponential, g_22, tt2, f22', [], [], opts);


high_power = +3;%dBm;
low_power = -15;%dBm;
powers = ones(size(timestamps))*low_power;
powers(1:N_high_power_spectra) = high_power;
powers(N_high_power_spectra+N_low_power_spectra+1:2*N_high_power_spectra+N_low_power_spectra) = high_power;


tta_all(ii_num,:)=tta;
ttb_all(ii_num,:)=ttb;
f1a_all(ii_num,:)=f1a;
f1b_all(ii_num,:)=f1b;
fs(ii_num)=mean(f1a+f1b)-fc;
fs2(ii_num)=mean(f1a2+f1b2)-fc;

tta2_all(ii_num,:)=tta2;
ttb2_all(ii_num,:)=ttb2;
f1a2_all(ii_num,:)=f1a2;
f1b2_all(ii_num,:)=f1b2;
%pause

N_dsa=(f1a*1e6-(fc*1e6+fs(ii_num)*1e6)/2).^2-(fc*1e6-fs(ii_num)*1e6).^2/4;
N_dsb=(f1b*1e6-(fc*1e6+fs(ii_num)*1e6)/2).^2-(fc*1e6-fs(ii_num)*1e6).^2/4;
N_dsa_all(ii_num,:)=N_dsa;
N_dsb_all(ii_num,:)=N_dsb;

N_dsa2=(f1a2*1e6-(fc*1e6+fs2(ii_num)*1e6)/2).^2-(fc*1e6-fs2(ii_num)*1e6).^2/4;
N_dsb2=(f1b2*1e6-(fc*1e6+fs2(ii_num)*1e6)/2).^2-(fc*1e6-fs2(ii_num)*1e6).^2/4;
N_dsa2_all(ii_num,:)=N_dsa2;
N_dsb2_all(ii_num,:)=N_dsb2;
end

%%

expon_fun=@(A,B,x0,tau,x) A*(B-exp(-(x-x0)/tau))
fo_exp=fitoptions('Method','NonlinearLeastSquares', 'Startpoint',[7.5e14,1,75,15],'Upper',[Inf,Inf,Inf,Inf],'Lower',[0,-Inf,0,0]);
exp_ft=fittype(expon_fun,'options',fo_exp)

figure()
for ii =1:5
    N_dsa=N_dsa_all(ii,:);
    N_dsb=N_dsb_all(ii,:);
    tta=tta_all(ii,:);
    ttb=ttb_all(ii,:);
    [Na_f,Na_g]=fit(tta',N_dsa',exp_ft);
    fo_expb=fitoptions('Method','NonlinearLeastSquares', 'Startpoint',[7.5e14,0,75,Na_f.tau],'Upper',[Inf,Inf,Inf,Inf],'Lower',[0,-Inf,0,0]);
    exp_ft=fittype(expon_fun,'options',fo_exp);

    [Nb_f,Nb_g]=fit(ttb',N_dsb',exp_ft);
    A_a(ii)=Na_f.A;
    B_a(ii)=Na_f.B;
    x0_a(ii)=Na_f.x0;
    tau_a(ii)=Na_f.tau;
    A_b(ii)=Nb_f.A;
    B_b(ii)=Nb_f.B;
    x0_b(ii)=Nb_f.x0;
    tau_b(ii)=Nb_f.tau;
    tta1_all(ii,:)=tta-x0_a(ii);
    ttb1_all(ii,:)=ttb-x0_b(ii);
    
    [Nboth_f,Nboth_g]=fit([tta1_all(ii,:)';ttb1_all(ii,:)'],[N_dsa';N_dsb'],exp_ft);
    A_both(ii)=Nboth_f.A;
    B_both(ii)=Nboth_f.B;
    x0_both(ii)=Nboth_f.x0;
    tau_both(ii)=Nboth_f.tau;

    subplot(2,3,ii)
    plot(tta1_all(ii,:), N_dsa,'xb',ttb1_all(ii,:),N_dsb,'xr')
    hold on
    %plot(Na_f,'b')
    %plot(Nb_f,'r')
    plot(tta1_all(ii,:),expon_fun(Na_f.A,Na_f.B,Na_f.x0,Na_f.tau,tta),'b')
    plot(ttb1_all(ii,:),expon_fun(Nb_f.A,Nb_f.B,Nb_f.x0,Nb_f.tau,ttb),'r')

    plot(Nboth_f,'k')
    hold off
end
%%
[Na_all_f,Na_all_g]=fit([tta1_all(:)],[N_dsa_all(:)],exp_ft);
[Nb_all_f,Nb_all_g]=fit([ttb1_all(:)],[N_dsb_all(:)],exp_ft);
[Nboth_all_f,Nboth_all_g]=fit([tta1_all(:);ttb1_all(:)],[N_dsa_all(:);N_dsb_all(:)],exp_ft);
figure()
plot([tta1_all(:)],[N_dsa_all(:)],'xb')
hold on
plot([ttb1_all(:)],[N_dsb_all(:)],'xr')
plot(Na_all_f,'b')
plot(Nb_all_f,'r')
plot(Nboth_all_f,'k')
hold off
ylabel('N')
xlabel('t (s)')
%% THIS MAY BE WRONG, 
%only looks good for FF6 

expon_fun2=@(A,B,x0,tau,x) A*(B+exp(-(x-x0)/tau))
fo_exp2=fitoptions('Method','NonlinearLeastSquares', 'Startpoint',[6e14,0.737,17,4],'Upper',[Inf,Inf,Inf,Inf],'Lower',[0,-Inf,0,0]);
exp_ft2=fittype(expon_fun2,'options',fo_exp2)

figure()
for ii =1:5
    N_dsa2=N_dsa2_all(ii,:);
    N_dsb2=N_dsb2_all(ii,:);
    tta2=tta2_all(ii,:);
    ttb2=ttb2_all(ii,:);
    [Na2_f,Na2_g]=fit(tta2',N_dsa2',exp_ft2);
    %fo_expb=fitoptions('Method','NonlinearLeastSquares', 'Startpoint',[7.5e14,0,75,Na2_f.tau],'Upper',[Inf,Inf,Inf,Inf],'Lower',[0,-Inf,0,0]);
    exp_ft2=fittype(expon_fun2,'options',fo_exp2);

    [Nb2_f,Nb2_g]=fit(ttb2',N_dsb2',exp_ft2);
    A_a2(ii)=Na2_f.A;
    B_a2(ii)=Na2_f.B;
    x0_a2(ii)=Na2_f.x0;
    tau_a2(ii)=Na2_f.tau;
    A_b2(ii)=Nb2_f.A;
    B_b2(ii)=Nb2_f.B;
    x0_b2(ii)=Nb2_f.x0;
    tau_b2(ii)=Nb2_f.tau;
    tta21_all(ii,:)=tta2-x0_a2(ii);
    ttb21_all(ii,:)=ttb2-x0_b2(ii);
    
    [Nboth2_f,Nboth2_g]=fit([tta21_all(ii,:)';ttb21_all(ii,:)'],[N_dsa2';N_dsb2'],exp_ft2);
    A_both2(ii)=Nboth2_f.A;
    B_both2(ii)=Nboth2_f.B;
    x0_both2(ii)=Nboth2_f.x0;
    tau_both2(ii)=Nboth2_f.tau;

    subplot(2,3,ii)
    plot(tta21_all(ii,:), N_dsa2,'xb',ttb21_all(ii,:),N_dsb2,'xr')
    hold on
    %plot(Na_f,'b')
    %plot(Nb_f,'r')
    plot(tta21_all(ii,:),expon_fun2(Na2_f.A,Na2_f.B,Na2_f.x0,Na2_f.tau,tta2),'b')
    plot(ttb21_all(ii,:),expon_fun2(Nb2_f.A,Nb2_f.B,Nb2_f.x0,Nb2_f.tau,ttb2),'r')

    plot(Nboth2_f,'k')
    hold off
end

%%
N_FF6_a=expon_fun(A_both(5),B_both(5),x0_both(5),tau_both(5),tta_all(5,:)-x0_a(5));
N_FF6_b=expon_fun(A_both(5),B_both(5),x0_both(5),tau_both(5),ttb_all(5,:)-x0_b(5));

dsa_FF6=(fc*1e6+fs(5)*1e6)/2-sqrt((fc*1e6-fs(5)*1e6).^2/4 +N_FF6_a);
dsb_FF6=(fc*1e6+fs(5)*1e6)/2+sqrt((fc*1e6-fs(5)*1e6).^2/4 +N_FF6_b);

N_FF6_a2=expon_fun2(A_both2(5),B_both2(5),x0_both2(5),tau_both2(5),tta2_all(5,:)-x0_a2(5));
N_FF6_b2=expon_fun2(A_both2(5),B_both2(5),x0_both2(5),tau_both2(5),ttb2_all(5,:)-x0_b2(5));

dsa2_FF6=(fc*1e6+fs2(5)*1e6)/2-sqrt((fc*1e6-fs2(5)*1e6).^2/4 +N_FF6_a2);
dsb2_FF6=(fc*1e6+fs2(5)*1e6)/2+sqrt((fc*1e6-fs2(5)*1e6).^2/4 +N_FF6_b2);

figure()
pcolor(timestamps_c,freq, 10*log10(trans))
set(get(gca,'children'),'edgecolor','none')
xlabel('t (s)')
ylabel('f (MHz)')
h = colorbar;
set(get(h,'ylabel'),'string','S_{21} (dB)')
set(gcf,'name',filename)
title(strrep(filename,'_','\_'))
grid on
hold on
plot(tta_all(5,:),dsa_FF6/1e6,'r', 'Linewidth',2)
plot(ttb_all(5,:),dsb_FF6/1e6,'r', 'Linewidth',2)
plot(tta2_all(5,:),dsa2_FF6/1e6,'r', 'Linewidth',2)
plot(ttb2_all(5,:),dsb2_FF6/1e6,'r', 'Linewidth',2)
fc_line=line([tt2(1),tt(end)],[fc,fc]);
set(fc_line,'color','b','Linewidth',1);
ylim([4650 4790])

