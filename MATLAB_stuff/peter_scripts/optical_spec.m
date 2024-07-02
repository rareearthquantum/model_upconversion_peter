wantplot=0

polyguassfun=@(mu,sd,A,mu1,C,B,x) B-C*exp(-(x-mu).^2/(2*sd^2)).*(A+(x-mu-mu1).^2);
polyguassfun2=@(mu,sd,C,C1,C2,C3,C4,A,x) A-C*exp(-(x-mu).^2/(2*sd^2)).*(1+C1*x+C2*x.^2+C3*x.^3+C4*x.^4);
gaussfun=@(mu,sd,C,B,x)B- C*exp(-(x-mu).^2/(2*sd^2));

dblorentz=@(C1,C2,gamsq1,gamsq2,x1,x2,B,x) B-(C1./((x-x1).^2+gamsq1)+C2./((x-x2).^2+gamsq2))
pg_fo=fitoptions('Method','NonlinearLeastSquares', 'Startpoint',[195.1177,0.00035,0.00000001,0.00045,0.5e7,0.97],'Upper',[195.12,0.001,1,Inf,Inf,0.99],'Lower',[195.115,0,0,-Inf,-Inf,0.6]);
pg_ft=fittype(polyguassfun,'options',pg_fo);
gauss_fo=fitoptions('Method','NonlinearLeastSquares', 'Startpoint',[195.11725,4e-4,0.8,0.97],'Upper',[195.12,0.001,Inf,0.99],'Lower',[195.115,0,-Inf,0.6]);
gauss_ft=fittype(gaussfun,'options',gauss_fo);

pg1_fo=fitoptions('Method','NonlinearLeastSquares', 'Startpoint',[195.11725,4e-4,0.8,0,0,0,0,0.97],'Upper',[195.12,0.001,Inf,Inf,Inf,Inf,Inf,0.99],'Lower',[195.115,0,-Inf,-Inf,-Inf,-Inf,-Inf,0.6]);
pg1_ft=fittype(polyguassfun2,'options',pg1_fo);

lorentz_fo=fitoptions('Method', 'NonlinearLeastSquares','Startpoint',[1.18e-7,6.6e-9,1.185e-07 ,1.666e-08,195.1171,195.1179,0.973],'Lower',[0,0,0,0,0,0,0.6],'Upper',[Inf,Inf,Inf,Inf,Inf,Inf,1]);
lorentz_ft=fittype(dblorentz,'options',lorentz_fo)
figure()
jj=1;
trans_min=zeros(1,54);
for ii =2:2:108
    
    lmin_ind=find(freq_opt(ii,:)>195.11627,1);
    freq_lmin=freq_opt(ii,lmin_ind);
 [trans_min(jj),min_ind]=min(opticaltrans(ii,:));
 %min_ind
 freq_min(jj)=freq_opt(ii,min_ind);
 r_ind=find(freq_opt(1,:)>195.1195,1);
 [~, l_ind]=max(opticaltrans(ii,lmin_ind:min_ind));
 l_ind=l_ind+lmin_ind+50;
 
 [pg_f,pg_g]=fit(freq_opt(ii,l_ind:r_ind)',opticaltrans(ii,l_ind:r_ind)',pg_ft);
 [gauss_f,gauss_g]=fit(freq_opt(ii,l_ind:r_ind)',opticaltrans(ii,l_ind:r_ind)',gauss_ft);
 [pg1_f,pg1_g]=fit(freq_opt(ii,l_ind:r_ind)',opticaltrans(ii,l_ind:r_ind)',pg1_ft);
     [lorentz_f,lorentz_g]=fit(freq_opt(ii,l_ind:r_ind)',opticaltrans(ii,l_ind:r_ind)',lorentz_ft);
sd_pg(jj)=pg_f.sd;
mu1_pg(jj)=pg_f.mu1;
mu_pg(jj)=pg_f.mu;
sd_gauss(jj)=gauss_f.sd;
mu_gauss(jj)=gauss_f.mu;
if wantplot
plot(freq_opt(ii,:),opticaltrans(ii,:))
hold on
plot(freq_opt(ii,l_ind:r_ind)',opticaltrans(ii,l_ind:r_ind)')
plot(pg_f)
%plot(pg1_f,'k')
plot(gauss_f,'b')
plot(lorentz_f,'k')
plot(freq_lmin,opticaltrans(ii,lmin_ind),'o')
plot(freq_opt(ii,l_ind),opticaltrans(ii,l_ind),'o')
%plot(freq_opt(ii,l_ind+50),opticaltrans(ii,l_ind+50),'o')

hold off
%xlim([195.115,195.119]) 
    pause
end
 jj=jj+1;
 
end
 
%%
polyguassfun=@(mu,sd,A,mu1,C,B,x) B-C*exp(-(x-mu).^2/(2*sd^2)).*(A+(x-mu-mu1).^2);
pg_fo=fitoptions('Method','NonlinearLeastSquares', 'Startpoint',[195.1177,0.00035,0.00000001,0.00045,0.5e7,0.98],'Upper',[195.12,0.001,1,Inf,Inf,1.1],'Lower',[195.115,0,0,-Inf,-Inf,0.6]);
pg_ft=fittype(polyguassfun,'options',pg_fo)


%%
figure()
subplot(4,1,1)
plot(fields(2:2:108),freq_min)
title('peak freq')
subplot(4,1,2)
plot(fields(2:2:108),mu_pg)
title('mu')
subplot(4,1,3)
plot(fields(2:2:108),mu1_pg)
title('mu1')
subplot(4,1,4)
plot(fields(2:2:108),sd_pg)
title('sd')
%%
figure()
subplot(3,1,1)
plot(fields(2:2:108),freq_min)
hold on
plot(fields(2:2:108),mu_gauss)
mu_poly=polyfit(fields(2:2:108),mu_gauss,1);
plot(fields(2:2:108),polyval(mu_poly,fields(2:2:108)))
hold off
title('peak freq')
subplot(3,1,2)
plot(fields(2:2:108),mu_gauss)
title('mu')
subplot(3,1,3)
plot(fields(2:2:108),sd_gauss)
sd_poly=polyfit(fields(2:2:108),sd_gauss,1);
hold on
plot(fields(2:2:108),polyval(sd_poly,fields(2:2:108)))
hold off
title('sd')