load ('05_CALTECH_FF1.notlargestmatrices.mat')

figure()
pcolor(X,fields,microwavetrans)
set(get(gca,'children'),'edgecolor','none')
xlabel('f (MHz)')
ylabel('I_{z} (A)')
title(filename,'interpreter','none')
set(gcf,'name',filename)
h = colorbar;
set(get(h,'ylabel'),'string','S_{21} (dB)')

%%
figure()
plot(X,mean(microwavetrans(1:250,:),1))
%%

lorentzian_fun=@(gammac,gammai,w0,x) gammac.^2./((gammac+gammai).^2/4+(x-w0).^2)
lorentzian_dBfun=@(gammac,gammai,w0,x) 10*log10(gammac.^2./((gammac+gammai).^2/4+(x-w0).^2))
fo1=fitoptions('Method','NonlinearLeastSquares', 'Startpoint',[0.1527,3.27,4733],'Upper',[Inf,Inf,Inf],'Lower',[0,0,0])
%ft1=fittype('I_fun1(Wsq,f0,a,x)','options',fo1)

lor_ft=fittype(lorentzian_fun,'options',fo1)
lordB_ft=fittype(lorentzian_dBfun,'options',fo1)

%f1=fit(f_max',fields', ft1)
[lor_f,lor_g]=fit(X',10.^(mean(microwavetrans(1:250,:),1)'/10), lor_ft)%,'weights',weight_fun(X(9:388)))
[lordB_f,lordB_g]=fit(X',(mean(microwavetrans(1:250,:),1)'), lordB_ft);%,'weights',weight_fun(X(9:388)))

figure()
hold on
plot(X,10.^(mean(microwavetrans(1:250,:),1)'/10))
plot(X,10.^(lordB_f(X)/10))
plot(X,(lor_f(X)))
legend('data','lor dB', 'lor')
hold off

figure()
hold on
plot(X,(mean(microwavetrans(1:250,:),1)))
plot(X,lordB_f(X))
hold off