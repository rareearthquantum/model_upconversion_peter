function t=timeslr(T);
A = 27;
B = 2.4e11;
delta = 83;
C = 1.4e-2;
r = A*T + C*T^9 + B*exp(-delta/T);
t=1/r;