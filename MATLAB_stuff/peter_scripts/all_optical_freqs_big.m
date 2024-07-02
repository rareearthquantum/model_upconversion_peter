opticaltrans=optitrans(2:2:end,2:10:end);
B_mag=B(2:2:end)*1e-3;
freq_opt= nu_mat(2:2:end,2:10:end)*1e-12;
figure()
pcolor(freq_opt, B_mag, opticaltrans)
set(get(gca,'children'),'edgecolor','none')
xlabel('\nu (THz)')
ylabel('B_{mag} (A)')
h = colorbar;
set(get(h,'ylabel'),'string','T (%)')

mask1=~(B_mag<0.14 | (B_mag>0.158&B_mag<0.164) | (B_mag>0.199&B_mag<0.2015));
%mask1=~(B_mag<0.14 | B_mag>0.2| (B_mag>0.158&B_mag<0.164));

B_mag=B_mag(mask1);
freq_opt=freq_opt(mask1,:);
opticaltrans=opticaltrans(mask1,:);

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
gaussfun=@(B,C1,C2,C3,C4,mu1,mu2,mu3,mu4,sd1,sd2,sd3,sd4,x) B-C1*exp(-(x-mu1).^2/(2*sd1^2))-C2*exp(-(x-mu2).^2/(2*sd2^2))-C3*exp(-(x-mu3).^2/(2*sd3^2))-C4*exp(-(x-mu4).^2/(2*sd4^2));
startpoints=[0.97,  0.1,0.3,0.9,0.6, 195.1125,195.11615,195.1175,195.120,  8e-4,4e-4,4e-4,4e-4  ];
up_bounds =[1.1, 1,1,1,1, 195.115,195.1166,195.1185,195.125,  Inf,Inf,Inf,Inf];
low_bounds=[0.6, 0,0,0,0, 195.110,195.1157,195.1168,195.119,   0,0,0,0];
gauss_fo=fitoptions('Method','NonlinearLeastSquares', 'Startpoint',startpoints,'Upper',up_bounds,'Lower',low_bounds);
gauss_ft=fittype(gaussfun,'options',gauss_fo);

jj=1;
trans_min=zeros(1,54);
clear mu1 mu2 mu3 mu4 C1 C2 C3 C4 sd1 sd2 sd3 sd4
for ii =1:length(B_mag)
   startpoints=[0.97,  0.1,0.3,0.9,0.6, polyval([-0.000213193072457,1.951167967376203]*1e2,B_mag(ii)),195.11615,195.1175,polyval([0.000214565838630,1.951166997158608]*1e2,B_mag(ii)),  8e-4,4e-4,4e-4,4e-4  ];

  gauss_fo=fitoptions('Method','NonlinearLeastSquares', 'Startpoint',startpoints,'Upper',up_bounds,'Lower',low_bounds);
gauss_ft=fittype(gaussfun,'options',gauss_fo);
  
    l_ind=find(freq_opt(ii,:)>195.105,1);  
 
 r_ind=find(freq_opt(1,:)>195.128,1);

 
 [gauss_f,gauss_g]=fit(freq_opt(ii,l_ind:r_ind)',opticaltrans(ii,l_ind:r_ind)',gauss_ft);
 [gauss_f,gauss_g]=fit(freq_opt(ii,:)',opticaltrans(ii,:)',gauss_ft);

mu1(jj)=gauss_f.mu1;
sd1(jj)=gauss_f.sd1;
C1(jj)=gauss_f.C1;

mu2(jj)=gauss_f.mu2;
sd2(jj)=gauss_f.sd2;
C2(jj)=gauss_f.C2;

mu3(jj)=gauss_f.mu3;
sd3(jj)=gauss_f.sd3;
C3(jj)=gauss_f.C3;

mu4(jj)=gauss_f.mu4;
sd4(jj)=gauss_f.sd4;
C4(jj)=gauss_f.C4;

if wantplot
plot(freq_opt(ii,:),opticaltrans(ii,:))
hold on
plot(freq_opt(ii,l_ind:r_ind)',opticaltrans(ii,l_ind:r_ind)')
%plot(pg1_f,'k')
plot(gauss_f,'b')
%plot(freq_opt(ii,l_ind+50),opticaltrans(ii,l_ind+50),'o')

hold off
%xlim([195.115,195.119]) 
    pause
end
 jj=jj+1;
 
end
 
%%

%%
figure()
plot(B_mag,mu1)
hold on
plot(B_mag,mu2) 
plot(B_mag,mu3)
plot(B_mag,mu4)
mu1_fit=polyfit(B_mag,mu1,1);
plot(B_mag,polyval(mu1_fit,B_mag))
mu2_fit=polyfit(B_mag,mu2,1);
plot(B_mag,polyval(mu2_fit,B_mag))
mu3_fit=polyfit(B_mag,mu3,1);
plot(B_mag,polyval(mu3_fit,B_mag))
mu4_fit=polyfit(B_mag,mu4,1);
plot(B_mag,polyval(mu4_fit,B_mag))
legend('1','2','3','4')
title('mu')
xlabel('B')
ylabel('mu THz')
hold off
%%
figure()
plot(B_mag,sd1)
hold on
plot(B_mag,sd2) 
plot(B_mag,sd3)
plot(B_mag,sd4)
sd1_fit=polyfit(B_mag,sd1,1);
plot(B_mag,polyval(sd1_fit,B_mag))
sd2_fit=polyfit(B_mag,sd2,1);
plot(B_mag,polyval(sd2_fit,B_mag))
sd3_fit=polyfit(B_mag,sd3,1);
plot(B_mag,polyval(sd3_fit,B_mag))
sd4_fit=polyfit(B_mag,sd4,1);
plot(B_mag,polyval(sd4_fit,B_mag))
legend('1','2','3','4')
title('sd')
xlabel('B')
ylabel('sd THz')
hold off

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
omega_mu=B_mag*24.1886e9*2*pi;
hbar=1.054e-34;
kB=1.38e-23
planck1=@(T,x) 1./(exp(omega_mu*hbar/kB/T)-1)
planck2=@(T,C0,x) C0./(exp(x*hbar/kB/T)-1)
planck3=@(T,C0,D,x) C0./(exp(x*hbar/kB/T)-1)+D

startpoints=[0.1,0.2];
up_bounds=[15,3];
low_bounds=[0,0];
planck_fo=fitoptions('Method','NonlinearLeastSquares', 'Startpoint',startpoints,'Upper',up_bounds,'Lower',low_bounds);
planck_ft=fittype(planck2,'options',planck_fo);
[planck_f1, planck_g1]=fit(omega_mu',C1(1:end)',planck_ft)
[planck_f2, planck_g2]=fit(omega_mu',C2(1:end)',planck_ft)
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