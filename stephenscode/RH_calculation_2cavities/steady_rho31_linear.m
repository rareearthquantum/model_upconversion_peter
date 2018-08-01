 function rho31_linear= steady_rho31_linear(omegao,omegam, delta3, delta2, ...
                                   params)

Rabimu=omegam;
Rabio=omegao;
xo = delta3;
xmu=delta2;

gamma13=params.gamma13;
gamma23=params.gamma23;
gamma2d=params.gamma2d;
gamma3d=params.gamma3d;
nbath=params.nbath;
gammab=params.gammab;

c21=1i*xmu+gammab/2*(2*nbath+1)+gamma2d/2;
c31=1i*xo+gammab/2*nbath+gamma13/2+gamma23/2+gamma3d/2;
c32=1i*(xo-xmu)+gammab/2*(nbath+1)+gamma13/2+gamma23/2+gamma3d/2+gamma2d/2;
%rho21_linear=(-1i)*Rabimu/(2*nbath+1)/c21;
%rho32_linear=(-1i)*Rabio*(nbath)/(2*nbath+1)/c32;        
rho31_linear=Rabio*Rabimu*(-1/(2*nbath+1)/c31/c21+(nbath)/(2*nbath+1)/c32/c31);
        


