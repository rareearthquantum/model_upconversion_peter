function [Rabio,Rabim,gk,Er_N,Eph]=initialization(Popt,source,kappao_c,kappao_a,p)
% kappao_c = 7.95e6*2*pi; % copuling loss of the cavity, inMHz
% kappao_a = 1.7e6*2*pi; % absroptive loss, in MHz
% Popt = 1e-3;
% source =0;
%% constants

epsilon0=8.854187817e-12; % electric constant, in m^-3 * kg^-1 * s^4 * A^2
mu0=4*pi*1e-7; % magnetic constant, in N * A^-2
hbar=1.05457e-34; % in J*s
NA = 6.0221413e23;    % avagadros number
kb = 1.3806488e-23;   % Boltzmann constant, J/K


fmw = 5.186e9;
f31 = 195113.36e9;
omega31 = f31*2*pi;
N_in = Popt/(hbar*omega31);

% optical laser       
wopt=1e-3;  % in m, the waist of our laser beam
Sspot=pi*wopt^2; % in m^2, we took the laser as a plane wave

 %% parameters of the system
n=1.76; % refreactive index
dipoleele=2e-32; % dipole moment, in C*m
d31=sqrt(2/3)*dipoleele/2;
d32=(sqrt(1/3))*dipoleele/2;

T=4.2; % temperature, in K
nbath=1/(exp(hbar*5.184e9*2*pi/(kb*T))-1); %Calculate the Thermal average photon number for the microwave bath

%% Calculate the number of Er atoms
density_YSO = 4.48e3;     % kg*m^-3
molecular_weight_YSO = 285.9;   % g/mol 
Er_faction=0.001e-2;
Lsample=12e-3; % the length of the sample, in m
dsample=5e-3; % the diameter of the sample, in m
Vsample=pi*(dsample/2)^2*Lsample; % the volume of the sample, which is the same as the volume of the microwave field inside the cavity
Er_N = density_YSO*Vsample/molecular_weight_YSO*1e3*2*Er_faction*NA;
Er_N =Er_N/2*nbath/(2*nbath+1); % two crystalgraphycal sides


%% Calculate the Rabi frequency of driving field inside the optical cavity
% eng_in = Popt * kappao_c*4/(kappao_c+kappao_a)^2;
n_in = N_in * kappao_c*4/(kappao_c+kappao_a)^2;

l_cav = (49.5+1.76*12)*1e-3;
V_cav = (Sspot*49.5e-3+Sspot*12e-3*1.76^3)/2;
Eopt = sqrt(n_in*hbar*omega31/2/epsilon0/V_cav);
Rabio=(d32*Eopt)/hbar*(-1);

hbar*omega31;
gk = d31*sqrt(hbar*omega31/(2*epsilon0*V_cav))/hbar;
Eph = sqrt(hbar*omega31/(2*epsilon0*V_cav));


% microwave
Q=5000; % Q factor
S21=-16; % in dB 
%% Calculate the Rabi frequency of microwave
S21=10^(S21/10);
Pmw = 1e-3 * 10^(source/10);
Engcav=sqrt(S21)*2/(2*pi*fmw/Q)*Pmw; % energy inside the microwave cavity, in J
Vsample=pi*(dsample/2)^2*Lsample; % the volume of the sample, which is the same as the volume of the microwave field inside the cavity
Bmw=sqrt(mu0*(Engcav/Vsample)/2); % Magnetic field of the microwave
mu21=p.mu21;
Rabim=(mu21*Bmw)/hbar*(-1); % in Hz








