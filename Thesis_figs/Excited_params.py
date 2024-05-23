from math import pi, sqrt


p={}
p['freqmu']=4733e6 #this is the microwave cavity frequency

laser_temp=38.6489
# laser_temp=38.6

p['freq_pump']=(-4.6118*laser_temp+195294.967)*1e9
# p['freq_pump'] = 195.11741e12
# p['freq_pump']=  1.951170437560000e+14

p['Gg']=0.0241886e12
p['Ge']=20.736275583971921e9
p['Gg']=19.795649223207793e9 #
p['Gg']=20.917352488350662e9

p['Ge']=0.020275e12
p['Ge']=20.287e9
#p['Gg']=0.0241886e12

p['f0_no_B']=195.1167943776907e12
p['d13'] = 2e-32*sqrt(2/3) #same spin
p['d12'] = 2e-32*sqrt(1/3) #different spin
p['gamma13'] = p['d13']**2/(p['d13']**2+p['d12']**2)*1/11e-3#*10
p['gamma12'] = p['d12']**2/(p['d13']**2+p['d12']**2)*1/11e-3#*10

p['gamma12'] = p['d13']**2/(p['d13']**2+p['d12']**2)*1/11e-3#*10
p['gamma13'] = p['d12']**2/(p['d13']**2+p['d12']**2)*1/11e-3#*10

# p['gamma12'] = 2*pi*0.1e6
# p['gamma13'] = 2*pi*0.1e6


p['gamma2d'] = 1e6
p['gamma3d'] = 1e6
#p['nbath'] = 20
p['gamma23'] = 1/11#1/(p['nbath']+1) * 1e3
# p['gamma23'] = 1/(1.5+1) * 1e4

# p['sd_delam']=2*pi*2e6
# p['sd_delam']=2*pi*25e6/2.355/2
p['sd_delam']=2*pi*2e6

p['gammaoc']=2*pi*1.7e6
p['gammaoi']=2*pi*7.95e6
p['gammamc']=2*pi*0.1522e6
p['gammami']=2*pi*3.255e6

p['gammamc']=2*pi*1.24e6
p['gammami']=2*pi*0.9399e6


muBohr=927.4009994e-26; # Bohr magneton in J/T in J* T^-1
p['mu12'] = 4.3803*muBohr # transition dipole moment for microwave cavity (J T^-1)


p['go'] = 51.9  #optical coupling

#p['No_total'] = 2.2e15 # number of atoms in the optical mode
#p['Nm_total'] = 6e16  #toal number of atoms
#p['Nm_total'] = 2e16  #toal number of atoms
p['No_total'] = 2.2e15 # number of atoms in the optical mode
p['Nm_total'] = 6e16*0.8  #toal number of atoms
# p['No'] = p['Nm'] # number of atoms in the optical mode

p['gm'] = 1.04/10 #coupling between atoms and microwave field
p['gm'] = 1.04*(1.2/4.3) #coupling between atoms and microwave field
p['Wbeam']=0.6e-3

p['Lsample']=12e-3 # the length of the sample, in m
p['Lcavity_vac'] = 49.5e-3 # length of the vacuum part of the optical Fabry Perot (m)
p['nYSO'] = 1.76 #refractive index of YSO

def omegaao1_from_B(B_mag,p):
    return (p['f0_no_B']+(-p['Gg']-p['Ge'])/2*B_mag)*2*pi
def omegaao2_from_B(B_mag,p):
    return (p['f0_no_B']+(-p['Gg']+p['Ge'])/2*B_mag)*2*pi
def omegaao3_from_B(B_mag,p):
    return (p['f0_no_B']+(+p['Gg']-p['Ge'])/2*B_mag)*2*pi
def omegaao4_from_B(B_mag,p):
    return (p['f0_no_B']+(+p['Gg']+p['Ge'])/2*B_mag)*2*pi

def sd_delao1_from_B(B_mag,p):
    return 1e9*(0.148588212918272+0.308180352441052*B_mag)*pi*2
def sd_delao2_from_B(B_mag,p):
    #return 1e9*(0.096660692060221-0.336895927831593*B_mag)*pi*2
    return 1e9*(0.096660692060221+0.336895927831593*B_mag)*pi*2
def sd_delao3_from_B(B_mag,p):
    return 1e9*(0.391366413926165+0.137444967951503*B_mag)*pi*2
def sd_delao4_from_B(B_mag,p):
    return 1e9*(0.279915072366011+0.000852680155581*B_mag)*pi*2


def sd_delao_from_B(B_mag,p):
    return sd_delao1_from_B(B_mag,p)

def deltaao_from_B(B_mag,deltamval,p):
    #return -(2*pi*(p['freqmu']+p['freq_pump'])-omegaa3o_from_B(B_mag,p))
    return omegaao1_from_B(B_mag,p)-(2*pi*p['freq_pump']-2*pi*p['freqmu']-deltamval)

#print(deltaao_from_B(0.233,0, p))
print('Importing Excited Params')
print(p)
