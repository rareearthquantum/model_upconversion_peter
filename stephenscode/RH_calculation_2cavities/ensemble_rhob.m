function [integ1,integ2,integ3] = ensemble_rhob(Rabio,Rabim,Rabisb,p)

% Rabio=0e6;Rabim =0e6;
sigo=p.sigmo;
sigm=p.sigmm;

%% work out scale factors required for the peak in the integrand to have a width of about 1

% d2max =  20*max([p.gamma2d,Rabim,Rabio]);
% d3max =  20*max([p.gamma3d,Rabio,Rabim]);%omega 
% if d2max>5*sigm;
%     d2max=5*sigm;
% end 
% if d3max>5*sigo;
%     d3max=5*sigo;
% end

scalex = sigo/17;
scaley = sigm/2.5;
scalez = 1/1e-10;
%% define function to be integrated
scaled = @(x,y)  steady_rho31(Rabio,Rabim,Rabisb,x*scalex,y*scaley,p);
% scaled = @(x,y) scalez*real((arrayfun(@(a,b) 1/(sqrt(2*pi)*sigm) * 1/(sqrt(2*pi)*sigo)* exp(-0.5*(a/sigo)^2-0.5*(b/sigm)^2),x*scalex,y*scaley)));

% transformed = @(t,s) scaled(t./(1-t.^2),s./(1-s.^2)).*(1+t.^2)./(1-t.^2).^2.*(1+s.^2)./(1-s.^2).^2;
transformed = @(t,s) scaled(t./(1-t.^2)/2,s./(1-s.^2)/2);



%% integrate integrate
   eps = 1e-8;
    n =p.n_int;
%  t = linspace(-1+eps,1-eps,n);
%  s = linspace(-1+eps,1-eps,n);

    [t,s] = meshgrid(linspace(-1+eps,1-eps,n),linspace(-1+eps,1-eps,n));
    z1 = t*0;
    z2 = t*0;
    z3 = t*0; 
    dt = t(2,2)-t(1,1);
    ds = s(2,2)-s(1,2);



parfor mm=1:1:numel(t)
%         warning('off')
        [temp1,temp2,temp3] = transformed(t(mm),s(mm));
        z1(mm) = scalez*temp1*(1+t(mm)^2)/(1-t(mm)^2)^2/2*(1+s(mm)^2)/(1-s(mm)^2)^2/2;
        z2(mm) = scalez*temp2*(1+t(mm)^2)/(1-t(mm)^2)^2/2*(1+s(mm)^2)/(1-s(mm)^2)^2/2; 
        z3(mm) = scalez*temp3*(1+t(mm)^2)/(1-t(mm)^2)^2/2*(1+s(mm)^2)/(1-s(mm)^2)^2/2; 
%          warning('on')
end

    
%     z1 = z1.*(1+t.^2)./(1-t.^2).^2.*(1+s.^2)./(1-s.^2).^2;
%     z2 = z2.*(1+t.^2)./(1-t.^2).^2.*(1+s.^2)./(1-s.^2).^2; 

%     make simpson weghts
%     w = simpsonw2(size(z1));
%     figure(110);
%     mesh(t,s,real(z1));shading flat;
%     xlabel('t')
%     ylabel('s')
    w = p.w;
    integ1 = sum(sum(w.*z1))*ds*dt *scalex*scaley/scalez;
    integ2 = sum(sum(w.*z2))*ds*dt *scalex*scaley/scalez;
    integ3 = sum(sum(w.*z3))*ds*dt *scalex*scaley/scalez;
% warning('on')
