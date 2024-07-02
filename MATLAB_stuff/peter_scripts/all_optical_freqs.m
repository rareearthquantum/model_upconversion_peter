wantplot=0

gaussfun=@(mu,sd,C,B,x)B- C*exp(-(x-mu).^2/(2*sd^2));
gaussfun=@(B,C1,C2,C3,C4,mu1,mu2,mu3,mu4,sd1,sd2,sd3,sd4,x) B-C1*exp(-(x-mu1).^2/(2*sd1^2))-C2*exp(-(x-mu2).^2/(2*sd2^2))-C3*exp(-(x-mu3).^2/(2*sd3^2))-C4*exp(-(x-mu4).^2/(2*sd4^2));
startpoints=[0.97,  0.1,0.3,0.9,0.6, 195.112,195.115,195.1175,195.122,  8e-4,4e-4,4e-4,4e-4  ];
up_bounds=[0.99, 1,1,1,1, 195.114,195.1166,195.119,195.123,  Inf,Inf,Inf,Inf]
low_bounds=[0.6, 0,0,0,0, 195.111,195.1157,195.117,195.12,   0,0,0,0];
gauss_fo=fitoptions('Method','NonlinearLeastSquares', 'Startpoint',startpoints,'Upper',up_bounds,'Lower',low_bounds);
gauss_ft=fittype(gaussfun,'options',gauss_fo);

jj=1;
trans_min=zeros(1,54);
for ii =2:2:108
    
    l_ind=find(freq_opt(ii,:)>195.11,1);  
 
 r_ind=find(freq_opt(1,:)>195.125,1);

 
 [gauss_f,gauss_g]=fit(freq_opt(ii,l_ind:r_ind)',opticaltrans(ii,l_ind:r_ind)',gauss_ft);

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

%%
figure()
plot(fields(2:2:108),mu1)
hold on
plot(fields(2:2:108),mu2) 
plot(fields(2:2:108),mu3)
plot(fields(2:2:108),mu4)
mu1_fit=polyfit(fields(2:2:108),mu1,1);
plot(fields(2:2:108),polyval(mu1_fit,fields(2:2:108)))
mu2_fit=polyfit(fields(2:2:108),mu2,1);
plot(fields(2:2:108),polyval(mu2_fit,fields(2:2:108)))
mu3_fit=polyfit(fields(2:2:108),mu3,1);
plot(fields(2:2:108),polyval(mu3_fit,fields(2:2:108)))
mu4_fit=polyfit(fields(2:2:108-12),mu4(1:end-6),1);
plot(fields(2:2:108-12),polyval(mu4_fit,fields(2:2:108-12)))
legend('1,','2','3','4')
hold off
figure()
plot(fields(2:2:108),sd1)
hold on
plot(fields(2:2:108),sd2)
plot(fields(2:2:108),sd3)
plot(fields(2:2:108),sd4)
sd1_fit=polyfit(fields(2:2:108-12),sd1(1:end-6),1);
plot(fields(2:2:108-12),polyval(sd1_fit,fields(2:2:108-12)))
sd2_fit=polyfit(fields(2:2:108-12),sd2(1:end-6),1);
plot(fields(2:2:108-12),polyval(sd2_fit,fields(2:2:108-12)))
sd3_fit=polyfit(fields(2:2:108-12),sd3(1:end-6),1);
plot(fields(2:2:108-12),polyval(sd3_fit,fields(2:2:108-12)))
sd4_fit=polyfit(fields(2:2:108-12),sd4(1:end-6),1);
plot(fields(2:2:108-12),polyval(sd4_fit,fields(2:2:108-12)))
legend('1,','2','3','4')
hold off
%%
figure()
plot(fields(2:2:108),C1)
hold on
plot(fields(2:2:108),C2)
plot(fields(2:2:108),C3)
plot(fields(2:2:108),C4)
C1_fit=polyfit(fields(2:2:108-12),C1(1:end-6),1);
plot(fields(2:2:108-12),polyval(C1_fit,fields(2:2:108-12)))
C2_fit=polyfit(fields(2:2:108-12),C2(1:end-6),1);
plot(fields(2:2:108-12),polyval(C2_fit,fields(2:2:108-12)))
C3_fit=polyfit(fields(2:2:108-12),C3(1:end-6),1);
plot(fields(2:2:108-12),polyval(C3_fit,fields(2:2:108-12)))
C4_fit=polyfit(fields(2:2:108-12),C4(1:end-6),1);
plot(fields(2:2:108-12),polyval(C4_fit,fields(2:2:108-12)))
legend('1,','2','3','4')
hold off