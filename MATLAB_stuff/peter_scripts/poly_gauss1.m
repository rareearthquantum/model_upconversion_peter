xx=linspace(195.115,195.119,1001);
mu=195.1177;
mu1=0.00045;
sd=0.00035;
yy=1-0.5e7*exp(-(xx-mu).^2/(2*sd^2)).*(-0.00000001+(xx-mu-mu1).^2);
figure()
plot(freq_opt(50,:),opticaltrans(50,:))
hold on
plot(xx,yy)
xlim([195.115,195.119])
hold off
