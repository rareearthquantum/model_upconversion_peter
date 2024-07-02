%params: fc,fs,gsq, tau,N
fc=4733e6
%fs=24.1886e9*200.2e-3
fs=(2*4729e6-fc)
%gsqN0 is g^2*N0, sqrt(gsqN0) is the unsaturated collective coupling
gsqN=@(gsqN0,tau,t) (1-exp(-t/tau))*gsqN0
dsa=@(gsqN0, tau,t0,x) (fc+fs)/2-sqrt((fc-fs).^2/4 +gsqN(gsqN0,tau,x-t0));
dsb=@(gsqN0, tau,t0,x) (fc+fs)/2+sqrt((fc-fs).^2/4 +(gsqN0*(1-exp(-(x-t0)/tau))));

dsa=@(gsqN0, tau,t0,x) (fc+fs)/2-sqrt((fc-fs).^2/4 +(gsqN0*(1-exp(-(x-t0)/tau))));
dsb=@(gsqN0, tau,t0,x) (fc+fs)/2+sqrt((fc-fs).^2/4 +(gsqN0*(1-exp(-(x-t0)/tau))));
%dsa=@(gsqN0,fs,fc, tau,t0,x) (fc+fs)/2-sqrt((fc-fs).^2/4 +gsqN0*(1-exp(-(x-t0)/tau)))
%dsb=@(gsqN0,fs,fc, tau,t0,x) (fc+fs)/2+sqrt((fc-fs).^2/4 +gsqN0*(1-exp(-(x-t0)/tau)))

%fo=fitoptions('Method','NonlinearLeastSquares', 'Startpoint',[1e15,fs,fc,10,82],'Upper',[Inf,Inf,Inf,Inf,810],'Lower',[0,0,0,0,10]);

fo=fitoptions('Method','NonlinearLeastSquares', 'Startpoint',[7.5e+14,20,82],'Upper',[1.2e15,Inf,810],'Lower',[0.3e15,0,70]);

dsa_ft = fittype(dsa, 'options', fo);
dsb_ft = fittype(dsb, 'options', fo);

[dsa_f, dsa_g]=fit(tta',f1a*1e6,dsa_ft);
[dsb_f, dsb_g]=fit(ttb',f1b*1e6,dsb_ft);
dsa_f
dsb_f

%%

figure()
hold on
plot(tta,f1a*1e6)
plot(dsa_f)
plot(ttb,f1b*1e6)
plot(dsb_f)
%%
N_dsa=(f1a*1e6-(fc+fs)/2).^2-(fc-fs).^2/4;
N_dsb=(f1b*1e6-(fc+fs)/2).^2-(fc-fs).^2/4;
expon_fun=@(A,B,x0,tau,x) A*(B-exp(-(x-x0)/tau))
fo_exp=fitoptions('Method','NonlinearLeastSquares', 'Startpoint',[7.5e14,1,75,10],'Upper',[Inf,Inf,Inf,Inf],'Lower',[0,-Inf,0,0]);
exp_ft=fittype(expon_fun,'options',fo_exp)
[Na_f,Na_g]=fit(tta',N_dsa,exp_ft);
[Nb_f,Nb_g]=fit(ttb',N_dsb,exp_ft);
[Nboth_f,Nboth_g]=fit([ttb,tta]',[N_dsb;N_dsa],exp_ft);

Na_f
Nb_f
figure()
hold on
plot(tta,N_dsa)
plot(Na_f)
plot(ttb,N_dsb)
plot(Nb_f)
plot(Nboth_f)

figure()
hold on
plot(tta,N_dsa)
plot(ttb,N_dsb)
plot(Nboth_f)
% 
% %%
% 
% dsa=@(gsqN0, tau1,tau2,t0,B,x) (fc+fs)/2-sqrt((fc-fs).^2/4 +(gsqN0*(1-B*exp(-(x-t0)/tau1) +(1-B)*exp(-(x-t0)/tau2) )))
% dsb=@(gsqN0, tau1,tau2,t0,B,x) (fc+fs)/2+sqrt((fc-fs).^2/4 +(gsqN0*(1-B*exp(-(x-t0)/tau1) +(1-B)*exp(-(x-t0)/tau2))))
% fo=fitoptions('Method','NonlinearLeastSquares', 'Startpoint',[0.5e15,20,200,82,0.95],'Upper',[1.2e15,Inf,Inf,810,1],'Lower',[0.3e15,0,0,70,0]);
% 
% dsa_ft = fittype(dsa, 'options', fo);
% dsb_ft = fittype(dsb, 'options', fo);
% 
% [dsa_f, dsa_g]=fit(tta',f1a*1e6,dsa_ft);
% [dsb_f, dsb_g]=fit(ttb',f1b*1e6,dsb_ft);
% dsa_f
% dsb_f
% figure()
% hold on
% plot(tta,f1a*1e6)
% plot(dsa_f)
% plot(ttb,f1b*1e6)
% plot(dsb_f)