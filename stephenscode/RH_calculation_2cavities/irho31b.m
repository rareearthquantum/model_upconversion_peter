function integ = irho31b(Rabio,Rabim,p)

sigo=p.sigmo;
sigm=p.sigmm;

%% work out scale factors required for the peak in the integrand to have a width of about 1

d2max =  20*max([p.gamma2d,Rabim,Rabio]);
d3max =  20*max([p.gamma3d,Rabio,Rabim]);%omega

if d2max>5*sigm;
    d2max=5*sigm;
end

if d3max>5*sigo;
    d3max=5*sigo;
end

scaley = d2max/5;
scalex = d3max/5;
%% work out scale factor required for the peak in the integrand to be about 1
scalez = 2*pi/steady_rho31_linear(Rabio,Rabim,0,0,p);

 

%% define function to be integrated

% scaled = @(x,y) ((arrayfun(@(a,b) steady_rho31(Rabio,Rabim,a,b,p),x*scalex,y*scaley)));
% scaled = @(x,y) scalez*((arrayfun(@(a,b) steady_rho31(Rabio,Rabim,a,b,p),x*scalex,y*scaley)));

scaled = @(x,y) scalez*real((arrayfun(@(a,b) 1/(sqrt(2*pi)*sigm) * 1/(sqrt(2*pi)*sigo)* exp(-0.5*(a/sigo)^2-0.5*(b/sigm)^2)*steady_rho31b(Rabio,Rabim,a,b,p),x*scalex,y*scaley)));

transformed = @(t,s) scaled(t./(1-t.^2),s./(1-s.^2)).*(1+t.^2)./(1-t.^2).^2.*(1+s.^2)./(1-s.^2).^2;
% transformed = @(t,s) scaled(t./(1-t.^2),s./(1-s.^2));



%% integrate integrate

   eps = 1e-8;
    n =p.n_int;
 
    [t,s] = meshgrid(linspace(-1+eps,1-eps,n),linspace(-1+eps,1-eps,n));
    dt = t(2,2)-t(1,1);
    ds = s(2,2)-s(1,1);
    warning('off')
    z = transformed(t,s);
    warning('on')
   
%     make simpson weghts
    w = simpsonw2(size(s));
    figure(111);
    mesh(t,s,real(z));shading flat;
    xlabel('t')
    ylabel('s')
    
    integ = sum(sum(w.*z))*ds*dt *scalex*scaley/scalez;


