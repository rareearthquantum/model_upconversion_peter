import numpy as np
import steady
import matplotlib.pyplot as plt

delta2vals = np.linspace(-10e6,10e6,1000)
delta3 = 0

p = {}
p['gamma_13'] = 1e6
p['gamma_23'] = 1e6
p['gamma_2d'] = 1e7
p['gamma_3d'] = 1e7
p['n_b'] = 0.0
p['gamma_m'] = 0e3
Omega_m = 0.0
Omega_o = 400e3
Omega_sb = 100e3


rho31 = 1j*delta2vals
rho31 = rho31*0

for (i, delta2) in enumerate(delta2vals):
    rho31[i] = steady.steady_rho31(Omega_o, Omega_m, Omega_sb,
                                   delta3, delta2, p)


plt.ion()
plt.plot(delta2vals, np.real(rho31), delta2vals, np.imag(rho31))
plt.show()
