opticaltrans_all=optitrans(2:2:end,2:10:end);
B_mag_all=B(2:2:end)*1e-3;
freq_opt_all= nu_mat(2:2:end,2:10:end)*1e-12;
figure()
pcolor(freq_opt_all, B_mag_all, opticaltrans_all)
set(get(gca,'children'),'edgecolor','none')
xlabel('\nu (THz)')
ylabel('B_{mag} (A)')
h = colorbar;
set(get(h,'ylabel'),'string','T (%)')

mask1=~(B_mag_all<0.14 | (B_mag_all>0.158&B_mag_all<0.164) | (B_mag_all>0.199&B_mag_all<0.2015));
%mask1=~(B_mag<0.14 | B_mag>0.2| (B_mag>0.158&B_mag<0.164));

B_mag=B_mag_all(mask1);
freq_opt=freq_opt_all(mask1,:);
opticaltrans=opticaltrans_all(mask1,:);

figure()
pcolor(freq_opt, B_mag, opticaltrans)
set(get(gca,'children'),'edgecolor','none')
xlabel('\nu (THz)')
ylabel('B_{mag} (A)')
h = colorbar;
set(get(h,'ylabel'),'string','T (%)')
%%
figure()
plot(freq_opt(1,:),opticaltrans(1,:))
hold on

plot(freq_opt(end,:),opticaltrans(end,:))
%plot(freq_opt(30,:),opticaltrans(30,:))
legend('1','end','mid')
hold off

%%
wantplot=0

gaussfun=@(mu,sd,C,B,x)B- C*exp(-(x-mu).^2/(2*sd^2));
%gaussfun=@(B,f0,Gg,Ge,C1,C2,C3,C4,sd1,sd2,sd3,sd4,x) B-C1*exp(-(x-(f0+())).^2/(2*sd1^2)...
    
startpoints=[0.97, 195.1167,0.022,0.021 0.1,0.3,0.9,0.6, 195.1125,195.11615,195.1175,195.120,  8e-4,4e-4,4e-4,4e-4  ];
up_bounds =[1.1, 195.1171, Inf,Inf, 1,1,1,1,  Inf,Inf,Inf,Inf];
low_bounds=[0.6, 195.1164, 0,0,     0,0,0,0,  0,0,0,0];
gauss_fo=fitoptions('Method','NonlinearLeastSquares', 'Startpoint',startpoints,'Upper',up_bounds,'Lower',low_bounds);
gauss_ft=fittype(gaussfun,'options',gauss_fo);

jj=1;
trans_min=zeros(1,54);
clear f0 Gg Ge C1 C2 C3 C4 sd1 sd2 sd3 sd4
for ii =1:length(B_mag)
    gaussfun=@(B,f0,Gg,Ge,C1,C2,C3,C4,sd1,sd2,sd3,sd4,x) B-C1*exp(-(x-(f0+(-Ge-Gg)/2*B_mag(ii))).^2/(2*sd1^2))...
                                                          -C2*exp(-(x-(f0+(+Ge-Gg)/2*B_mag(ii))).^2/(2*sd2^2))...
                                                          -C3*exp(-(x-(f0+(-Ge+Gg)/2*B_mag(ii))).^2/(2*sd3^2))...
                                                          -C4*exp(-(x-(f0+(+Ge+Gg)/2*B_mag(ii))).^2/(2*sd4^2));

startpoints=[0.97, 195.1167,0.022,0.021 0.1,0.3,0.9,0.6, 8e-4,4e-4,4e-4,4e-4  ];

  gauss_fo=fitoptions('Method','NonlinearLeastSquares', 'Startpoint',startpoints,'Upper',up_bounds,'Lower',low_bounds);
gauss_ft=fittype(gaussfun,'options',gauss_fo);
  
    l_ind=find(freq_opt(ii,:)>195.105,1);  
 
 r_ind=find(freq_opt(1,:)>195.128,1);

 
 [gauss_f,gauss_g]=fit(freq_opt(ii,l_ind:r_ind)',opticaltrans(ii,l_ind:r_ind)',gauss_ft);
 [gauss_f,gauss_g]=fit(freq_opt(ii,:)',opticaltrans(ii,:)',gauss_ft);

f0(ii)=gauss_f.f0;
Gg(ii)=gauss_f.Gg;
Ge(ii)=gauss_f.Ge;
sd1(jj)=gauss_f.sd1;
C1(jj)=gauss_f.C1;

sd2(jj)=gauss_f.sd2;
C2(jj)=gauss_f.C2;

sd3(jj)=gauss_f.sd3;
C3(jj)=gauss_f.C3;

sd4(jj)=gauss_f.sd4;
C4(jj)=gauss_f.C4;

% if wantplot & mod(ii,2)==0
if ii==43
plot(freq_opt(ii,:),opticaltrans(ii,:))
hold on
plot(freq_opt(ii,l_ind:r_ind)',opticaltrans(ii,l_ind:r_ind)')
%plot(pg1_f,'k')
plot(gauss_f,'b')
%plot(freq_opt(ii,l_ind+50),opticaltrans(ii,l_ind+50),'o')
freq_opt_sl=freq_opt(ii,:);
gauss_f_sl=gauss_f;
opticaltrans_sl=opticaltrans(ii,:);
B_mag_sl=B_mag(ii);
hold off
%xlim([195.115,195.119]) 
%     pause(0.1)
end
 jj=jj+1;
 
end
 
%%

figure()
i1=30;
hold on
plot(freq_opt(i1,:),opticaltrans(i1,:))
plot(freq_opt(i1,:),gaussfun(B_mag(i1),f0(i1),Gg(i1),Ge(i1),C1(i1),C2(i1),C3(i1),C4(i1),sd1(i1),sd2(i1),sd3(i1),sd4(i1),opticaltrans(i1,:)))
hold off
disp('hi')
%%
%%
figure()
plot(B_mag,Gg)
hold on
plot(B_mag,Ge) 
Gg_fit=polyfit(B_mag,Gg,1);
plot(B_mag,polyval(Gg_fit,B_mag))
Ge_fit=polyfit(B_mag,Ge,1);
plot(B_mag,polyval(Ge_fit,B_mag))
legend('G_g','G_e')
%title('mu')
xlabel('B (T)')
ylabel('g factor THz')
hold off
%%
figure()
plot(B_mag,sd1*1e3,'-','linewidth',2)
hold on
plot(B_mag,sd2*1e3,'-','linewidth',2) 
plot(B_mag,sd3*1e3,'-','linewidth',2)
plot(B_mag,sd4*1e3,'-','linewidth',2)
sd1_fit=polyfit(B_mag,sd1,1);
plot(B_mag,polyval(sd1_fit,B_mag)*1e3,'linewidth',1.2,'color','black')
sd2_fit=polyfit(B_mag,sd2,1);
plot(B_mag,polyval(sd2_fit,B_mag)*1e3,'linewidth',1.2,'color','black')
sd3_fit=polyfit(B_mag,sd3,1);
plot(B_mag,polyval(sd3_fit,B_mag)*1e3,'linewidth',1.2,'color','black')
sd4_fit=polyfit(B_mag,sd4,1);
plot(B_mag,polyval(sd4_fit,B_mag)*1e3,'linewidth',1.2,'color','black')
leg=legend('1','2','3','4');
set(leg,'interpreter','latex','fontsize',14,'Position',[0.7786 0.6053 0.1077 0.2264])
%title('\sigma_o')
xlabel('$B$ (T)','interpreter','latex','fontsize',14)
ylabel('$\sigma_o$ (GHz)','interpreter','latex','fontsize',14)
hold off

h=gcf
set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(h,'optical_sigma1','-dpdf','-r0','-bestfit')
%%
figure()
plot(B_mag,f0)
hold on
f0_fit=polyfit(B_mag,f0,1);
plot(B_mag,polyval(f0_fit,B_mag))
title('f0')
xlabel('B')
ylabel('nu THz')
%%
figure()
plot(B_mag,C1)
hold on
plot(B_mag,C2) 
plot(B_mag,C3)
plot(B_mag,C4)
C1_fit=polyfit(B_mag,C1,1);
plot(B_mag,polyval(C1_fit,B_mag))
C2_fit=polyfit(B_mag,C2,1);
plot(B_mag,polyval(C2_fit,B_mag))
C3_fit=polyfit(B_mag,C3,1);
plot(B_mag,polyval(C3_fit,B_mag))
C4_fit=polyfit(B_mag,C4,1);
plot(B_mag,polyval(C4_fit,B_mag))
legend('1','2','3','4')
title('abs')
xlabel('B')
ylabel('arb')
hold off

%%
Gg_mean=mean(Gg)
Ge_mean=mean(Ge)
f0_mean=mean(f0)
figure()
pcolor(freq_opt_all, B_mag_all, opticaltrans_all)
set(get(gca,'children'),'edgecolor','none')
hold on
% plot(f0_mean+(Ge_mean+Gg_mean)/2*B_mag,B_mag,'r')
% plot(f0_mean+(Ge_mean-Gg_mean)/2*B_mag,B_mag,'r')
% plot(f0_mean+(-Ge_mean+Gg_mean)/2*B_mag,B_mag,'r')
% plot(f0_mean+(-Ge_mean-Gg_mean)/2*B_mag,B_mag,'r')
% 
plot(f0_mean+(Ge_mean+Gg_mean)/2*B_mag_all,B_mag_all,'r','linewidth',2)
plot(f0_mean+(Ge_mean-Gg_mean)/2*B_mag_all,B_mag_all,'r','linewidth',2)
plot(f0_mean+(-Ge_mean+Gg_mean)/2*B_mag_all,B_mag_all,'r','linewidth',2)
plot(f0_mean+(-Ge_mean-Gg_mean)/2*B_mag_all,B_mag_all,'r','linewidth',2)
Ge2=0.020275
Gg2=0.0241886
plot(f0_mean+(Ge2+Gg2)/2*B_mag_all,B_mag_all,'b','linewidth',2)
plot(f0_mean+(Ge2-Gg2)/2*B_mag_all,B_mag_all,'b','linewidth',2)
plot(f0_mean+(-Ge2+Gg2)/2*B_mag_all,B_mag_all,'b','linewidth',2)
plot(f0_mean+(-Ge2-Gg2)/2*B_mag_all,B_mag_all,'b','linewidth',2)
xlabel('\nu (THz)')
ylabel('B (T)')
h = colorbar;
set(get(h,'ylabel'),'string','T (%)')
B_mag_slice=0.2328;
(f0_mean+(-Ge_mean-Gg_mean)/2*B_mag_slice)
(f0_mean+(Ge_mean-Gg_mean)/2*B_mag_slice)
(f0_mean+(-Ge_mean+Gg_mean)/2*B_mag_slice)
(f0_mean+(Ge_mean+Gg_mean)/2*B_mag_slice)

%%
mask2=~((B_mag_all>0.158&B_mag_all<0.164) | (B_mag_all>0.199&B_mag_all<0.2015));
%mask1=~(B_mag<0.14 | B_mag>0.2| (B_mag>0.158&B_mag<0.164));

B_mag2=B_mag_all(mask2);
freq_opt2=freq_opt_all(mask2,:);
opticaltrans2=opticaltrans_all(mask2,:);
figure()
pcolor(freq_opt2, B_mag2, opticaltrans2)
set(get(gca,'children'),'edgecolor','none')
hold on
% plot(f0_mean+(Ge_mean+Gg_mean)/2*B_mag,B_mag,'r')
% plot(f0_mean+(Ge_mean-Gg_mean)/2*B_mag,B_mag,'r')
% plot(f0_mean+(-Ge_mean+Gg_mean)/2*B_mag,B_mag,'r')
% plot(f0_mean+(-Ge_mean-Gg_mean)/2*B_mag,B_mag,'r')
% 
plot(f0_mean+(Ge_mean+Gg_mean)/2*B_mag_all,B_mag_all,'r','linewidth',2)
plot(f0_mean+(Ge_mean-Gg_mean)/2*B_mag_all,B_mag_all,'r','linewidth',2)
plot(f0_mean+(-Ge_mean+Gg_mean)/2*B_mag_all,B_mag_all,'r','linewidth',2)
plot(f0_mean+(-Ge_mean-Gg_mean)/2*B_mag_all,B_mag_all,'r','linewidth',2)
xlabel('Optical Frequency (THz)','interpreter','latex','fontsize',14)
ylabel('$B$ (T)','interpreter','latex','fontsize',14)
h = colorbar;
set(get(h,'ylabel'),'string','Transmission (\%)','interpreter','latex','fontsize',14)
xlim([195.105,195.127])
h=gcf
set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(h,'rhosmallb_ds_excited','-dpng','-r900')
%%
omega_mu=B_mag*24.1886e9*2*pi;
hbar=1.054e-34;
kB=1.38e-23;
planck1=@(T,x) 1./(exp(omega_mu*hbar/kB/T)-1);
planck2=@(T,C0,x) C0./(exp(x*hbar/kB/T)-1);
planck3=@(T,C0,D,x) C0./(exp(x*hbar/kB/T)-1)+D;

startpoints=[0.1,0.2];
up_bounds=[15,3];
low_bounds=[0,0];
planck_fo=fitoptions('Method','NonlinearLeastSquares', 'Startpoint',startpoints,'Upper',up_bounds,'Lower',low_bounds);
planck_ft=fittype(planck2,'options',planck_fo);
[planck_f1, planck_g1]=fit(omega_mu',C1(1:end)',planck_ft);
[planck_f2, planck_g2]=fit(omega_mu',C2(1:end)',planck_ft);
% startpoints=[0.1,0.5,0];
% up_bounds=[15,3,Inf];
% low_bounds=[0,0,-Inf];
% planck_fo=fitoptions('Method','NonlinearLeastSquares', 'Startpoint',startpoints,'Upper',up_bounds,'Lower',low_bounds);
% planck_ft=fittype(planck3,'options',planck_fo);
% [planck_f1, planck_g1]=fit(omega_mu',C1(1:end-6)',planck_ft)
% [planck_f2, planck_g2]=fit(omega_mu',C2(1:end-6)',planck_ft)

figure()
plot(omega_mu,C1(1:end))
hold on

plot(omega_mu,C2(1:end))

plot(planck_f1)
plot(planck_f2)
hold off
%%
omega_mu=B_mag*24.1886e9*2*pi;
hbar=1.054e-34;
kB=1.38e-23;
planck1=@(T,x) 1./(exp(omega_mu*hbar/kB/T)-1);
planck2=@(T,C0,x) C0./(exp(x*hbar/kB/T)-1);
planck3=@(T,C0,D,x) C0./(exp(x*hbar/kB/T)-1)+D;
A1=sqrt(2*pi)*C1.*sd1;
A2=sqrt(2*pi)*C2.*sd2;
startpoints=[0.1,0.2e-3];
up_bounds=[15,3];
low_bounds=[0,0];
planck_fo=fitoptions('Method','NonlinearLeastSquares', 'Startpoint',startpoints,'Upper',up_bounds,'Lower',low_bounds);
planck_ft=fittype(planck2,'options',planck_fo);
[planckA_f1, planckA_g1]=fit(omega_mu',A1(1:end)',planck_ft);
[planckA_f2, planckA_g2]=fit(omega_mu',A2(1:end)',planck_ft);
% startpoints=[0.1,0.5,0];
% up_bounds=[15,3,Inf];
% low_bounds=[0,0,-Inf];
% planck_fo=fitoptions('Method','NonlinearLeastSquares', 'Startpoint',startpoints,'Upper',up_bounds,'Lower',low_bounds);
% planck_ft=fittype(planck3,'options',planck_fo);
% [planck_f1, planck_g1]=fit(omega_mu',C1(1:end-6)',planck_ft)
% [planck_f2, planck_g2]=fit(omega_mu',C2(1:end-6)',planck_ft)

figure()
plot(omega_mu,A1(1:end))
hold on

plot(omega_mu,A2(1:end))

plot(planckA_f1)
plot(planckA_f2)
hold off
%%

planck1=@(T,x) 1./(exp(x*hbar/kB/T)-1);
%pop_eff=@(T,C0,D,x) C0*(1./(exp(x*hbar/kB/T)+1)+D);%C0./(2*planck1(T,x)+1);
pop_eff=@(T,C0,D,x) C0*(1./(exp(x*hbar/kB/T)+1)+D);%C0./(2*planck1(T,x)+1);

A1=sqrt(2*pi)*C1.*sd1(1);
A2=sqrt(2*pi)*C2.*sd2(1);
startpoints=[0.1,2e-4,1e-6];
up_bounds=[15,3,0.0001];
low_bounds=[0,0,0];
pop_fo=fitoptions('Method','NonlinearLeastSquares', 'Startpoint',startpoints,'Upper',up_bounds,'Lower',low_bounds);
pop_ft=fittype(pop_eff,'options',pop_fo);
[popA_f1, popA_g1]=fit(omega_mu',A1(1:end)',pop_ft)

startpoints=[0.14,9.5e-4,1e-6];
up_bounds=[1,3,0.01];
low_bounds=[0,0,0];
pop_fo=fitoptions('Method','NonlinearLeastSquares', 'Startpoint',startpoints,'Upper',up_bounds,'Lower',low_bounds);
pop_ft=fittype(pop_eff,'options',pop_fo);
[popA_f2, popA_g2]=fit(omega_mu',A2(1:end)',pop_ft)
% startpoints=[0.1,0.5,0];
% up_bounds=[15,3,Inf];
% low_bounds=[0,0,-Inf];
% planck_fo=fitoptions('Method','NonlinearLeastSquares', 'Startpoint',startpoints,'Upper',up_bounds,'Lower',low_bounds);
% planck_ft=fittype(planck3,'options',planck_fo);
% [planck_f1, planck_g1]=fit(omega_mu',C1(1:end-6)',planck_ft)
% [planck_f2, planck_g2]=fit(omega_mu',C2(1:end-6)',planck_ft)

figure()
plot(omega_mu,A1(1:end))
hold on

plot(omega_mu,A2(1:end))

plot(popA_f1)
plot(popA_f2)
xlim([0,5e10])
hold off

%%

planck1=@(T,x) 1./(exp(x*hbar/kB/T)-1);
%pop_eff=@(T,C0,D,x) C0*(1./(exp(x*hbar/kB/T)+1)+D);%C0./(2*planck1(T,x)+1);
pop_eff=@(T,C0,D,x) C0*(1./(exp(x*hbar/kB/T)+1)+D);%C0./(2*planck1(T,x)+1);

A1=C1;
A2=C2;
startpoints=[0.1,0.34,1e-6];
up_bounds=[15,3,0.0001];
low_bounds=[0,0,0];
pop_fo=fitoptions('Method','NonlinearLeastSquares', 'Startpoint',startpoints,'Upper',up_bounds,'Lower',low_bounds);
pop_ft=fittype(pop_eff,'options',pop_fo);
[popA_f1, popA_g1]=fit(omega_mu',A1(1:end)',pop_ft)

startpoints=[0.12,2,1e-6];
up_bounds=[1,3,1];
low_bounds=[0,0,-1];
pop_fo=fitoptions('Method','NonlinearLeastSquares', 'Startpoint',startpoints,'Upper',up_bounds,'Lower',low_bounds);
pop_ft=fittype(pop_eff,'options',pop_fo);
[popA_f2, popA_g2]=fit(omega_mu',A2(1:end)',pop_ft)
% startpoints=[0.1,0.5,0];
% up_bounds=[15,3,Inf];
% low_bounds=[0,0,-Inf];
% planck_fo=fitoptions('Method','NonlinearLeastSquares', 'Startpoint',startpoints,'Upper',up_bounds,'Lower',low_bounds);
% planck_ft=fittype(planck3,'options',planck_fo);
% [planck_f1, planck_g1]=fit(omega_mu',C1(1:end-6)',planck_ft)
% [planck_f2, planck_g2]=fit(omega_mu',C2(1:end-6)',planck_ft)

figure()
plot(omega_mu,A1(1:end))
hold on

plot(omega_mu,A2(1:end))

plot(popA_f1)
plot(popA_f2)
xlim([0,5e10])
hold off
%%

figure()
plot(freq_opt_sl,opticaltrans_sl,'b','linewidth',1.5)
hold on
plot(freq_opt_sl,gauss_f_sl(freq_opt_sl),'r','linewidth',2)
hold off
xlabel('Optical Frequency (THz)','interpreter','latex','fontsize',14)
ylabel('Transmission','interpreter','latex','fontsize',14)
xlim([195.105,195.127])
ylim([0,1.07])
h=gcf
set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(h,'optitrans_slice1','-dpdf')
