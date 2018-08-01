function [a,Aout,Pout,b,generation,absorption] = cal_aout(Popt,source,aold,bold,p);
misfit = 1;
count = 1;
delta_laser = 0*1e6*2*pi;
delta_mw = 0*1e6*2*pi;
kappao_c = 7.95e6*2*pi; % copuling loss of the cavity, inMHz
kappao_a = 1.7e6*2*pi; % absroptive loss, in MHz
kappam_c=70e3*2*pi;
kappam_a=650e3*2*pi;
emw = 3.4363e-24; % energy per mw photon
Pmw = 1e-3 * 10^(source/10);

[Rabio,Rabim,gk,uk,Er_No,Er_Nm]=initialization(Popt,source,kappao_c,kappao_a,kappam_c,kappam_a,p);
Rabio;
Rabim;
bnew=0;

while misfit>1e-4
    if count==1;
        aold = aold*1.2;   
        bold = bold*1.2;
    else
        aold = anew;
        bold = bnew;
    end
    [z1,z2,z3] = ensemble_rhob(Rabio,bold*uk,aold*gk,p);
%      [z1,z2,z3] = ensemble_rhob(Rabio,Rabim,aold*gk,p);
%     temp1 = z1*(-1i)*gk*Er_N
    irho31new = z1*Er_No;
    idiff31new = z2*Er_No; 
    irho21new = z3*Er_Nm;
    anew = (-1i*gk*(irho31new-idiff31new)) /(-1i*delta_laser+(kappao_c+kappao_a)/2 -(-1i*gk*idiff31new/aold));
    bnew = (-sqrt(kappam_c*Pmw/emw)) /(-1i*delta_mw+(kappam_c*2+kappam_a)/2 -(-1i*uk*irho21new/bold));
%         bnew = (-sqrt(kappam_c*Pmw/emw)) /(-1i*delta_mw+(kappam_c*2+kappam_a)/2);

    Rabim = bnew*uk
    (-1i*uk*irho21new/bold)
    absorption = -1i*gk*idiff31new/aold;
    generation = irho31new-idiff31new;
    Aoutnew = sqrt(kappao_c)*anew;    
    
    misfit = abs((abs(anew))^2/(abs(aold))^2-1)+abs((abs(bnew))^2/(abs(bold))^2-1)
%      misfit = abs((abs(anew))^2/(abs(aold))^2-1)
    
    count = count+1;
end
a = anew;
b = bnew;
Aout = Aoutnew;
Pout = Aout^2*1.29e-19
misfit;
