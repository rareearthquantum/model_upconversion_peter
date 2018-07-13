function params = getparams(d31,d32,gamma3d,gamma2d)
% d31, optical transition strength between levle 1 and 3; d32, optical transition strength between levle 2 and 3;
% gamma2d, dephasing time of level 2; gamma3d, dephasing of level 3.


muB=927.4e-26; % in J* T^-1
T=4.2; % temperature, in K
fmu=5.186e9; % resonant frequency of the cavity, in Hz;
omegamu=2*pi*fmu; 
hbar = 1.0545717e-34;
kb=1.3806488e-23 ;
nbath=1/(exp(hbar*omegamu/(kb*T))-1); %Calculate the Thermal average photon number for the microwave bath
%nbath=20;


ltime2=(1e-3); % life time of level 2, assuming the life time is 100 ms;
ltime3=(11e-3); % decay rate of State 3, assuming the life time is 11 ms;
gammab=1/ltime2*1/(nbath+1)/2;

sigmm=10*1e6*2*pi; % in Hz, inhomogeneous broadening deviation of microwave transition;
sigmo=100*1e6*2*pi; % in Hz, inhomogeneous broadening deviation of microwave transition;

%% Calculation of mu12, d31, and d32.
thita=34;    % in degree, the angel between the DC B field and the D1 axis of YSO
thita=thita*pi/180; 
% dipoleele=2e-32; % dipole moment, in C*m
[munorm,~,~]=transitions(thita);
mu21=abs(munorm(2,1))*muB;
% d31=abs(d31norm)*dipoleele;
% d32=abs(d32norm)*dipoleele;
%d32=1;
%d31=1*sqrt(2);
gamma13=1/ltime3*(d31^2/(d31^2+d32^2));
gamma23=1/ltime3*(d32^2/(d31^2+d32^2));



params.gamma13=gamma13;
params.gamma23=gamma23;
params.gammab=gammab;
params.gamma2d=gamma2d;
params.gamma3d=gamma3d;
params.nbath=nbath;
params.gammab=gammab;
params.ltime2=ltime2;
params.sigmm=sigmm; % in Hz, inhomogeneous broadening deviation of microwave transition;
params.sigmo=sigmo;
params.d31=d31;
params.d32=d32;
params.mu21=mu21;
params.n_int=421;
