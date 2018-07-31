function params = getparams(d31,d32,gamma3d,gamma2d)
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
gammab=1/ltime2*1/(2*nbath+1);

sigmm=25*1e6*2*pi; % in Hz, inhomogeneous broadening deviation of microwave transition;
sigmo=170*1e6*2*pi; % in Hz, inhomogeneous broadening deviation of microwave transition;

%% Calculation of mu12, d31, and d32.
thita=29;    % in degree, the angel between the DC B field and the D1 axis of YSO
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

%% make the collapse operators
s12 = qo([1 0 0 ]'*[0 1 0]);
s23 = qo([0 1 0 ]'*[0 0 1]);
s13=s12*s23;
s22 = s12'*s12;
s33 = s23'*s23;
s21 = s12';

collapse_operators = {};

collapse_operators{1} = sqrt(gamma13/2)*s13;
collapse_operators{2} = sqrt(gamma23/2)*s23;
collapse_operators{3} = sqrt(gammab/2*(nbath+1))*s12;
collapse_operators{4} = sqrt(gammab/2*nbath)*s21;
collapse_operators{5} = sqrt(gamma2d/2)*s22;
collapse_operators{6} = sqrt(gamma3d/2)*s33;

Lloss = linblad(collapse_operators{1});
for k=2:length(collapse_operators)
    Lloss=Lloss+linblad(collapse_operators{k});
end





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
n_int=4021;
params.n_int=n_int;
w = simpsonw2(size(zeros(n_int,n_int)));
params.w = w;
