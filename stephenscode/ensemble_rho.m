function [integ1,integ2] = ensemble_rho(Rabio,Rabim,Rabisb,p)

% Rabio=0e6;Rabim =0e6;
sigo=p.sigmo;   % the optical inhomo broadening
sigm=p.sigmm;   % the mw inhomo broadening

%% work out scale factors required for the peak in the integrand to have a width of about 1

% d2max =  20*max([p.gamma2d,Rabim,Rabio]);
% d3max =  20*max([p.gamma3d,Rabio,Rabim]);%omega 
% if d2max>5*sigm;
%     d2max=5*sigm;
% end 
% if d3max>5*sigo;
%     d3max=5*sigo;
% end

scalex = sigo/5;
scaley = sigm/5;

%% define function to be integrated
scaled = @(x,y) ((arrayfun(@(a,b) steady_rho31(Rabio,Rabim,Rabisb,a,b,p),x*scalex,y*scaley)));
% steay_rho31 is to calculate the steady states of different detunings

transformed = @(t,s) scaled(t./(1-t.^2),s./(1-s.^2));



%% integrate integrate
   eps = 1e-8;
    n =p.n_int;
 
    [t,s] = meshgrid(linspace(-1+eps,1-eps,n),linspace(-1+eps,1-eps,n));
    dt = t(2,2)-t(1,1);
    ds = s(2,2)-s(1,1);
    warning('off')
    [z1,z2] = transformed(t,s);
    warning('on')
    z1 = z1.*(1+t.^2)./(1-t.^2).^2.*(1+s.^2)./(1-s.^2).^2;
    z2 = z2.*(1+t.^2)./(1-t.^2).^2.*(1+s.^2)./(1-s.^2).^2; 

%     make simpson weghts
    w = simpsonw2(size(s));
%     figure(110);
%     mesh(t,s,real(z1));shading flat;
%     xlabel('t')
%     ylabel('s')
    
    integ1 = sum(sum(w.*z1))*ds*dt *scalex*scaley;  % uses simpson method to do integrations
    integ2 = sum(sum(w.*z2))*ds*dt *scalex*scaley;


