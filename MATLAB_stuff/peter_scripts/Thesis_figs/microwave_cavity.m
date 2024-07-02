%load('17_CALTECH_FF10.mat')


%lorentzian_fun=@(gammac,gammai,f0,x) gammac^2./((gammai+gammac).^2/4-(x-w0).^2)
%lorentzian_dBfun=@(gammac,gammai,f0,x) 10*log10(gammac^2./((gammai+gammac).^2/4-(x-w0).^2))
slice_ind=1;
lorentzian_fun=@(gammac,gammai,w0,x) gammac.^2./((gammac+gammai).^2/4+(x-w0).^2)
lorentzian_fun2=@(gammasq,w0,x) gammasq*max(10.^(microwavetrans(slice_ind,150:250)'/10))./4./(gammasq/4+(x-w0).^2);
lorentzian_dBfun=@(gammac,gammai,w0,x) 10*log10(gammac.^2./((gammac+gammai).^2/4+(x-w0).^2))
fo1=fitoptions('Method','NonlinearLeastSquares', 'Startpoint',[0.3,6,5017],'Upper',[Inf,Inf,Inf],'Lower',[0,0,0]);
fo2=fitoptions('Method','NonlinearLeastSquares', 'Startpoint',[36,5017],'Upper',[Inf,Inf],'Lower',[0,0]);

%ft1=fittype('I_fun1(Wsq,f0,a,x)','options',fo1)

lor_ft=fittype(lorentzian_fun,'options',fo1)
lor_ft2=fittype(lorentzian_fun2,'options',fo2)
lordB_ft=fittype(lorentzian_dBfun,'options',fo1)

%f1=fit(f_max',fields', ft1)
[lor_f,lor_g]=fit(X(150:250)',10.^(microwavetrans(slice_ind,150:250)'/10), lor_ft);%,'weights',weight_fun(X(9:388)))
[lor_f2,lor_g2]=fit(X(150:250)',10.^(microwavetrans(slice_ind,150:250)'/10), lor_ft2);%,'weights',weight_fun(X(9:388)))
[lordB_f,lordB_g]=fit(X(150:250)',(microwavetrans(slice_ind,150:250)'), lordB_ft);%,'weights',weight_fun(X(9:388)))
lor_f
lor_f2
lordB_f
% figure()
% hold on
% plot(X,10.^(microwavetrans(1,:)'/10))
% plot(X,10.^(lordB_f(X)/10))
% plot(X,(lor_f(X)))
% plot(X,(lor_f2(X)))
% 
% hold off

figure()
hold on
plot(X,(microwavetrans(1,:)))%,'.')
plot(X,lordB_f(X),'linewidth',2)
ylabel('Cavity Transmission (dB)')
xlabel('Frequency (MHz)')
legend('Experiment', 'Lorentzian Fit')