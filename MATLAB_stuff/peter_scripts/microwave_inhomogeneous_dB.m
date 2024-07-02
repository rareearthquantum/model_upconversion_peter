% field_min_ind=40
% field_max_ind=60
% gau_mean=[];
% gau_sd=[];
% 
% figure()
% gauss_lor_fun=@(A,gammasq,lor_mean,B,sdsq,gau_mean,x) A./(gammasq+(x-lor_mean).^2)+B*exp(-(x-gau_mean).^2/2/sdsq)
% for ii=field_min_ind:field_max_ind
% %     subplot(121)
% %     plot(X,microwavetrans(ii,:))
% %     subplot(122)
% %     plot(X,10.^(microwavetrans(ii,:)/10))
% %     
%     gau_mean_guess=(fields(ii)-7.6453)*2*340.1011+5019;
%     fo=fitoptions('Method','NonlinearLeastSquares', 'Startpoint',[1e-4,6,5015,1e-5,6,gau_mean_guess],'Upper',[Inf,Inf,Inf,Inf,Inf,Inf],'Lower',[0,0,0,0,0,0]);
%     guass_lor_ft=fittype(gauss_lor_fun,'options',fo);
%     [gl_f,gl_g]=fit(X', 10.^(microwavetrans(ii,:)'/10),guass_lor_ft);
%     gau_mean=[gau_mean,gl_f.gau_mean];
%     gau_sd=[gau_sd,sqrt(gl_f.sdsq)];
%     hold on
% 	plot(X,10.^(microwavetrans(ii,:)/10))
%     plot(gl_f)
%     hold off
%     pause
%     
%     cla
% end
microwavetrans1=padarray(microwavetrans,[6 6],0,'both');
X1=[X(1:6)-X(8)+X(2) X X(end-5:end)+X(7)-X(1)];
X_inds1=1:185;
X_inds2=215:401;
field_inds1=46:64;
field_inds2=39:57;

X_inds1=1:185;
X_inds2=215:401;
field_inds1=5:38;
field_inds2=2:29;
fields=fliplr(I_z)

% X_inds1=6:191;
% X_inds2=221:407;
% field_inds1=52:72;
% field_inds2=43:63;
aa=340.1011;
bb=7.6453;
Wsq=1.4120e3;
f0=5019;
I_fun2=@(I) aa*(I-bb)+sqrt(Wsq+aa^2*(I-bb).^2)+f0;
I_fun1=@(I) aa*(I-bb)-sqrt(Wsq+aa^2*(I-bb).^2)+f0;
gauss_fun_log=@(A,mu,sd,x) 10*log10(A*exp(-(x-mu).^2/2/sd.^2));
gauss_fun=@(A,mu,sd,x) (A*exp(-(x-mu).^2/2/sd.^2));

lor_fun=@(A,X0,gam,x)A*gam^2./(gam.^2+(x-X0).^2);
figure()
sd_fwhm1=[];
sd_fwhm2=[];
mean_fwhm1=[];
mean_fwhm2=[];
A_fwhm1=[];
A_fwhm2=[];
for ii =field_inds1
    %ii
    mean_guess=I_fun1(fields(ii));
    A_guess=max(10.^(microwavetrans(ii,X_inds1)'/10));
    A_guess2=10.^(microwavetrans(ii,find(X>mean_guess,1))'/10);
    [A_guess3, A_guess3_ind]=max(10.^(microwavetrans(ii,max([find(X>mean_guess,1)-5,2]):find(X>mean_guess,1)+5)'/10));
    A_guess3_ind=A_guess3_ind+max([find(X>mean_guess,1)-5,2])-1;
    mean_guess3=X(A_guess3_ind);
    HM_ind1=find((10.^(microwavetrans(ii,:)/10))>A_guess3/2&(X<mean_guess3),1);
    HM_ind2=find((10.^(microwavetrans(ii,:)/10))<A_guess3/2&(X>mean_guess3),1);
    fwhm=X(HM_ind2)-X(HM_ind1);
    mean_fwhm=(X(HM_ind2)+X(HM_ind1))/2;
    sd_fwhm=fwhm/2.355;
    sd_fwhm1=[sd_fwhm1,sd_fwhm]
    mean_fwhm1=[mean_fwhm1,mean_fwhm];
    A_fwhm1=[A_fwhm1,A_guess3];
%     fo=fitoptions('Method','NonlinearLeastSquares', 'Startpoint',[A_guess2,mean_guess,1],'Upper',[Inf,Inf,Inf],'Lower',[0,0,0]);
%     gauss_ft=fittype(gauss_fun,'options',fo);
%     [gauss_f, gauss_g]=fit(X(X_inds1)',10.^(microwavetrans(ii,X_inds1)'/10),gauss_ft,'weights',gauss_fun(1, mean_guess,15 ,X(X_inds1)'));
%     
    fo=fitoptions('Method','NonlinearLeastSquares', 'Startpoint',[A_guess2,mean_guess,1],'Upper',[Inf,Inf,Inf],'Lower',[0,0,0]);
    gauss_ft=fittype(gauss_fun,'options',fo);
    [gauss_f, gauss_g]=fit(X(X_inds1)',(microwavetrans(ii,X_inds1)'),gauss_ft,'weights',gauss_fun(1, mean_guess,15 ,X(X_inds1)'));
    
    fo_lor=fitoptions('Method','NonlinearLeastSquares', 'Startpoint',[A_guess2,mean_guess,1],'Upper',[Inf,Inf,Inf],'Lower',[0,0,0]);
    gauss_ft=fittype(lor_fun,'options',fo_lor);
    [lor_f, lor_g]=fit(X(X_inds1)',10.^(microwavetrans(ii,X_inds1)'/10),gauss_ft,'weights',gauss_fun(1, mean_guess,15 ,X(X_inds1)'));
    
    hold on
    plot(X(X_inds1),(microwavetrans(ii,X_inds1)'))
    xlims=get(gca,'XLim');
    ylims=get(gca,'YLim');
    
    %plot(10*log10(gauss_f),'r')
    %plot(lor_f,'g')
    %plot(mean_guess,A_guess,'o')
    %plot(mean_guess,A_guess2,'o') 
    %plot(mean_guess3,A_guess3,'o')
    %plot(X([HM_ind1,HM_ind2]),10.^(microwavetrans(ii,[HM_ind1,HM_ind2])'/10),'.-')
    %plot(X(X_inds2),gauss_fun(A_guess3,mean_guess3,sd_fwhm,X(X_inds2)))
    plot(X(X_inds1),10*log10(gauss_fun(A_guess3,mean_fwhm,sd_fwhm,X(X_inds1))))
    %plot(X(X_inds1),(gauss_fun_log(A_guess3,mean_fwhm,sd_fwhm,X(X_inds1))))

    xlim(xlims);
    ylim(ylims);
    %plot(X,(((10.^(microwavetrans(ii,:)'/10))<A_guess3/2)&(X<mean_guess3) )*1e-4)
    hold off
    pause
    cla
    
end
%%

figure()
for ii =field_inds2
    mean_guess=I_fun2(fields(ii));
    A_guess=max(10.^(microwavetrans(ii,X_inds2)/10));
    A_guess2=10.^(microwavetrans(ii,find(X>mean_guess,1))/10);
    [A_guess3, A_guess3_ind]=max(10.^(microwavetrans(ii,find(X>mean_guess,1)-5:min([find(X>mean_guess,1)+5,length(microwavetrans)-5]))/10));
    A_guess3_ind=A_guess3_ind+find(X>mean_guess,1)-6;
    mean_guess3=X(A_guess3_ind);
    HM_ind1=find((10.^(microwavetrans(ii,:)/10))>A_guess3/2&(X<mean_guess3&X>5019),1,'last')-1 ;
    HM_ind2=find((10.^(microwavetrans(ii,:)/10))<A_guess3/2&(X>mean_guess3),1) ;
    fwhm=X(HM_ind2)-X(HM_ind1);
    mean_fwhm=(X(HM_ind2)+X(HM_ind1))/2;
    sd_fwhm=fwhm/2.355;
    sd_fwhm2=[sd_fwhm2,sd_fwhm]
    mean_fwhm2=[mean_fwhm2 ,mean_fwhm];
    A_fwhm2=[A_fwhm2,A_guess3];

    fo=fitoptions('Method','NonlinearLeastSquares', 'Startpoint',[A_guess2,mean_guess,1],'Upper',[Inf,Inf,Inf],'Lower',[0,0,0]);
    gauss_ft=fittype(gauss_fun,'options',fo);
    [gauss_f, gauss_g]=fit(X(X_inds2)',10.^(microwavetrans(ii,X_inds2)'/10),gauss_ft,'weights',gauss_fun(1, mean_guess,15 ,X(X_inds2)'));
    
    fo_lor=fitoptions('Method','NonlinearLeastSquares', 'Startpoint',[A_guess2,mean_guess,1],'Upper',[Inf,Inf,Inf],'Lower',[0,0,0]);
    gauss_ft=fittype(lor_fun,'options',fo_lor);
    [lor_f, lor_g]=fit(X(X_inds2)',10.^(microwavetrans(ii,X_inds2)'/10),gauss_ft,'weights',gauss_fun(1, mean_guess,15 ,X(X_inds2)'));
    
    hold on
    plot(X(X_inds2),(microwavetrans(ii,X_inds2)'))
    xlims=get(gca,'XLim');
    ylims=get(gca,'YLim'); 
    
    
    plot(X(X_inds2),10*log10(gauss_fun(A_guess3,mean_fwhm,sd_fwhm,X(X_inds2))))
    xlim(xlims)
    ylim(ylims)
    hold off
    pause
    cla
    
end


