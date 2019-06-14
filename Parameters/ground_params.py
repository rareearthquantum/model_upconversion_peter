import numpy as np
from math import pi
import math
def Omega_from_PdBm(PdBm,p):
    mu0=4*pi*1e-7
    c=3e8
    hbar=1.05457e-34; # in J*s
    P=1e-3*10**(PdBm/10)
    Abeam=pi*p['Wbeam']**2/4
    Efield=np.sqrt(2*mu0*c*P/Abeam)
    Omega=p['d23']*Efield/hbar
    return Omega
def b_from_Pin(Pin,p):
    mu0=4*pi*1e-7
    hbar=1.05457e-34; # in J*s
    Vsample=pi*((p['dsample']/2)**2) * p['Lsample'] # the volume of the Er:YSO sample
    V_microwave_cavity=Vsample/p['fillfactor']
    mwP=1e-3*10**(Pin/10)
    Q=2*pi*p['freqmu']/(p['gammamc']+p['gammami'])
    S12=(p['gammamc']**2)/(p['gammamc']+p['gammami'])**2
    E_cavity=np.sqrt(S12)*2/(2*pi*p['freqmu']/Q)*mwP
    Bwm=np.sqrt(mu0*(E_cavity/V_microwave_cavity)/2)
    b=(p['mu12']*Bwm)/hbar
    return b

def bin_from_PdBm(Pdbm,p,deltamval=0):
    P=1e-3*10**(Pdbm/10)
    hbar=1.05457e-34; # in J*s
    omega=2*np.pi*(p['freqmu'])+deltamval
    bin=np.sqrt(P/hbar/omega)
    return bin
def nbath_from_T(T,p,deltamacval=0):
    omega=np.abs(2*pi*p['freqmu']-deltamacval)
    nbath = 1/(np.exp(1.0545718e-34*omega/1.38064852e-23/T)-1)
    return nbath
P_pump = 10*np.log10(1.74) #in dBm, 1.74 mW are going into the resonator
P_mu = -15 -30 # in dBm
T=100e-3
p={}
p['freqmu']=4733e6 #this is the microwave cavity frequency
p['freq_pump'] = 195113.36e9 #pump frequency
p['freqo']=p['freqmu']+p['freq_pump']

p['nbath']=nbath_from_T(T,p)
p['d13'] = 2e-32*math.sqrt(1/3)
p['d23'] = 2e-32*math.sqrt(2/3)
p['gamma13'] = p['d13']**2/(p['d13']**2+p['d23']**2)*1/11e-3
p['gamma23'] = p['d23']**2/(p['d13']**2+p['d23']**2)*1/11e-3
p['gamma2d'] = 1e6
p['gamma3d'] = 1e6
#p['nbath'] = 20
p['gammamu'] = 1/(p['nbath']+1) * 1e3

p['go'] = 51.9  #optical coupling

p['No'] = 1.28e15 # number of atoms in the optical mode


p['Nm'] = 2.22e16  #toal number of atoms
p['gm'] = 1.04 #coupling between atoms and microwave field

p['Wbeam']=0.6e-3



p['Omega']= Omega_from_PdBm(P_pump,p)

print('Omega = ' + str(Omega_from_PdBm(P_pump,p)))
print('bin   = ' + str(bin_from_PdBm(P_mu,p)))
