II=fields(2:2:108-12);
A=0.027684;
B=0.056331;
B=(A*II*1e3-B)*1e-3;
omega_mu=B*24.1886e9*2*pi;
hbar=1.054e-34;
kB=1.38e-23
planck1=@(T,x) 1./(exp(omega_mu*hbar/kB/T)-1)
planck2=@(T,C0,x) C0./(exp(x*hbar/kB/T)-1)
planck3=@(T,C0,D,x) C0./(exp(x*hbar/kB/T)-1)+D

startpoints=[0.1,0.5];
up_bounds=[15,3];
low_bounds=[0,0];
planck_fo=fitoptions('Method','NonlinearLeastSquares', 'Startpoint',startpoints,'Upper',up_bounds,'Lower',low_bounds);
planck_ft=fittype(planck2,'options',planck_fo);
[planck_f1, planck_g1]=fit(omega_mu',C1(1:end-6)',planck_ft)
[planck_f2, planck_g2]=fit(omega_mu',C2(1:end-6)',planck_ft)
% startpoints=[0.1,0.5,0];
% up_bounds=[15,3,Inf];
% low_bounds=[0,0,-Inf];
% planck_fo=fitoptions('Method','NonlinearLeastSquares', 'Startpoint',startpoints,'Upper',up_bounds,'Lower',low_bounds);
% planck_ft=fittype(planck3,'options',planck_fo);
% [planck_f1, planck_g1]=fit(omega_mu',C1(1:end-6)',planck_ft)
% [planck_f2, planck_g2]=fit(omega_mu',C2(1:end-6)',planck_ft)

figure()
plot(omega_mu,C1(1:end-6))
hold on

plot(omega_mu,C2(1:end-6))

plot(planck_f1)
plot(planck_f2)
hold off