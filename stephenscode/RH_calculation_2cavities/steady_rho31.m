 function [rho31,diff31,rho21] = steady_rho31(Rabio,Rabim, Rabisb, delta3, delta2, ...
                                   params)

Rabio=Rabio;
Rabimu=Rabim;
A=Rabisb;
xo = delta3+delta2;
xmu=delta2;

gamma13=params.gamma13;
gamma23=params.gamma23;
gamma2d=params.gamma2d;
gamma3d=params.gamma3d;
nbath=params.nbath;
gammab=params.gammab;
sigmo = params.sigmo;
sigmm = params.sigmm;
% scalez = params.scalez;


M=zeros(9,9);

% elements that are independent on detunings
M(1,1)=1;       M(1,5)=1;       M(1,9)=1;
M(2,1)=1i*Rabimu';	M(2,3)=1i*Rabio;    M(2,5)=-1i*Rabimu'; M(2,8)=-1i*A';
M(3,1)=1i*A';   M(3,2)=1i*Rabio';	M(3,6)=-1i*Rabimu'; M(3,9)=-1i*A';	
M(4,1)=-1i*Rabimu;	M(4,5)=1i*Rabimu;	M(4,6)=1i*A;    M(4,7)=-1i*Rabio';
M(5,1)=gammab*nbath;	M(5,2)=-1i*Rabimu;	M(5,4)=1i*Rabimu';	M(5,5)=-gammab*(nbath+1);	M(5,6)=1i*Rabio;    M(5,8)=-1i*Rabio';   M(5,9)=gamma23;
M(6,3)=-1i*Rabimu;	M(6,4)=1i*A';   M(6,5)=1i*Rabio';	M(6,9)=-1i*Rabio';
M(7,1)=-1i*A;   M(7,4)=-1i*Rabio;	M(7,8)=1i*Rabimu;   M(7,9)=1i*A; 
M(8,2)=-1i*A;   M(8,5)=-1i*Rabio;	M(8,7)=1i*Rabimu';   M(8,9)=1i*Rabio;
M(9,3)=-1i*A;   M(9,6)=-1i*Rabio;   M(9,7)=1i*A';   M(9,8)=1i*Rabio';    M(9,9)=-(gamma13+gamma23);

% elements that vary with mw detuning
M(2,2)=-gammab/2*(2*nbath+1)+1i*xmu-gamma2d/2;
M(4,4)=-gammab/2*(2*nbath+1)-1i*xmu-gamma2d/2;
       
% elements that vary with optical detuning
M(3,3)=-(gammab*nbath+gamma13+gamma23)/2+1i*xo-gamma3d/2;
M(6,6)=-(gammab*(nbath+1)+gamma13+gamma23)/2+1i*(xo-xmu)-gamma3d/2-gamma2d/2;
M(7,7)=-(gammab*nbath+gamma13+gamma23)/2-1i*xo-gamma3d/2;
M(8,8)=-(gammab*(nbath+1)+gamma13+gamma23)/2-1i*(xo-xmu)-gamma3d/2-gamma2d/2;        
                                             
     
A=[1;0;0;0;0;0;0;0;0];                         
rho=M\A;

%rho11 = rho(1);
coeff = 1/(sqrt(2*pi)*sigmm) * 1/(sqrt(2*pi)*sigmo)* exp(-0.5*(delta3/sigmo)^2-0.5*(delta2/sigmm)^2);
rho31 = rho(7)*coeff;
diff31 = (rho(1)-rho(9))/(1i*xo+(gammab*nbath+gamma13+gamma23)/2+gamma3d/2)*(-1i*Rabisb)*coeff;
rho21 = rho(4)*coeff;
% rho31 = 1*coeff;
% diff31 = 1*coeff;


