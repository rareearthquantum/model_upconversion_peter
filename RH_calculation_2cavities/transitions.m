function [mu,d31,d32]=transitions(thita);

%% calculate the normalized microwave transition
% addpath('c:\qotoolbox');
gg  = [       3.07      -3.124       3.396 ;
            -3.124       8.156      -5.756;
             3.396      -5.756       5.787  ];    % in MHz, ground state g factor.

BDC=[cos(thita),sin(thita),0];  % normalized DC D field.
Sx=jmat(1/2,'x'); % spin matrices
Sy=jmat(1/2,'y');
Sz=jmat(1/2,'z');
BeffDCg=BDC*gg;
HgDC=BeffDCg(1)*Sx+BeffDCg(2)*Sy+BeffDCg(3)*Sz; % Hamiltonian of the ground state under the DC B field.
[estatesDCg,evaluesDCd] = simdiag(HgDC);

BRF=[0,0,1];
BeffRF=BRF*gg;
HRF=BeffRF(1)*Sx+BeffRF(2)*Sy+BeffRF(3)*Sz; % Hamiltonian of the AC B field
mu11=estatesDCg{1}'*HRF*estatesDCg{1};  % mormalized values
mu12=estatesDCg{1}'*HRF*estatesDCg{2};
mu21=estatesDCg{2}'*HRF*estatesDCg{1};
mu22=estatesDCg{2}'*HRF*estatesDCg{2};

mu=[double(mu11),double(mu12);double(mu21),double(mu22)]; 


%% calculate the normalized optical transition

ge  = [     1.950      -2.212       3.584 ;
            -2.212       4.232      -4.986;
             3.584      -4.986       7.888  ];
BeffDCe=BDC*ge;
HeDC=BeffDCe(1)*Sx+BeffDCe(2)*Sy+BeffDCe(3)*Sz;
[estatesDCe,evaluesDCe] = simdiag(HeDC); 


straa  =  double(estatesDCe{1}'*estatesDCg{1});  % normalized values
strab  =  double(estatesDCe{1}'*estatesDCg{2}); 
strba  =  double(estatesDCe{2}'*estatesDCg{1}); 
strbb  =  double(estatesDCe{2}'*estatesDCg{2}); 

d31=strbb;
d32=strab;