function [a,Aout,Pout] = cal_aout(Popt,source,aold,p);
misfit = 1;
count = 1;
delta_laser = 000*1e6*2*pi;
kappao_c = 7.95e6*2*pi; % the copuling loss of the optical cavity, in MHz
kappao_a = 1.7e6*2*pi; % the absroptive loss oftheoptical cavity, in MHz

[Rabio,Rabim,gk,Er_N,Eph]=initialization(Popt,source,kappao_c,kappao_a,p); % gets a trial set of parameters for an empty optical cavity
Rabio
Rabim

while misfit>1e-4
    if count==1;
        aold = aold*1.05;  % the step factor 1.05 should be changed according to the step of the calculated mw.      
    else
        aold = anew;
    end
    [z1,z2] = ensemble_rho(Rabio,Rabim,aold*gk,p);  % calculates the integration of the density matrix over opt and mw detunings
%     temp1 = z1*(-1i)*gk*Er_N
    irho31new = z1*Er_N;
    idiff31new = z2*Er_N; 
    anew = (-1i*gk*(irho31new-idiff31new)) /(-1i*delta_laser+(kappao_c+kappao_a)/2 -(-1i*gk*idiff31new/aold));  % uses iteration to calculate a new
%     tolook = -1i*gk*idiff31new/aold
    Aoutnew = sqrt(kappao_c)*anew;
    misfit = abs((abs(anew))^2/(abs(aold))^2-1) % defines a misfit for the iteration
    count = count+1;
end
a = anew;
Aout = Aoutnew;
Pout = Aout^2*1.29e-19 % 1.29e-19 is the energy per photon at 1536 nm

