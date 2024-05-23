import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sys
sys.path.append('/home/peter/model_upconversion')

#import qutip

from matplotlib.colors import Normalize as Norm


class MidpointNormalize(colors.Normalize):
	"""
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))




filename='Thesis_figs/Compare_models/Adiabat_sim3T = 0.05, P_mu = -70 '
npzfile=np.load(filename+'.npz')
print(npzfile)
p=npzfile['p'][()]
bvals=npzfile['bvals']#/np.sqrt(p['gammamc'])
binvals = npzfile['binvals']
effic_abia=npzfile['effic_abia']
good_vals_abia=npzfile['good_vals_abia']
#delta2vals=npzfile['delta2vals']
delta2vals=np.linspace(-2000e6,2000e6,51)*2/10

deltamuvals=npzfile['deltamuvals']
Cba=npzfile['Cba']
avals=npzfile['avals']
effic_abia=npzfile['effic_abia']
effic_abia=npzfile['effic_abia']


effic_full=np.abs(avals*np.sqrt(p['gammaoc'])/binvals)**2

fig=plt.figure(filename+'effic_adiabat/effic_linear')

fig.clf()

ax=fig.add_subplot(3,1,1)
#img1=ax.imshow(10*np.log10(np.abs(effic_abia)/np.abs(Cba**2)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower',cmap='RdBu')
img1=ax.imshow(np.log10(np.abs(effic_abia)/np.abs(Cba**2)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower',cmap='RdBu_r',norm=MidpointNormalize(midpoint=0))
#img1=ax.imshow(np.abs(10*np.log10(np.abs(effic_abia)/np.abs(Cba**2))),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
#img1=ax.imshow(10*np.log10(np.abs(effic_abia)/np.abs(Cba**2)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
#plt.plot(p['No']*p['gm']**2/delta2vals2,delta2vals2)
plt.title(' |C|^2 (dB scale)')
plt.ylabel('atom delta')
plt.xlabel('Cavity delta')
fig.colorbar(img1)

ax=fig.add_subplot(3,1,2)
img1=ax.imshow((np.abs(effic_abia)/np.abs(Cba**2)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
plt.title('|C|^2 ')
plt.ylabel('atom delta')
plt.xlabel('Cavity delta')
fig.colorbar(img1, format='%.1e')

ax=fig.add_subplot(3,1,3)
img1=ax.imshow(good_vals_abia,extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
plt.title('good?')
plt.ylabel('atom delta')
plt.xlabel('Cavity delta')
fig.colorbar(img1)

fig.suptitle('effic_adiabat/effic_linear')

fig=plt.figure(filename+'effic_adiabat/full')

fig.clf()

ax=fig.add_subplot(3,1,1)
#img1=ax.imshow(10*np.log10(np.abs(effic_abia)/np.abs(Cba**2)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower',cmap='RdBu')
img1=ax.imshow(np.log10(np.abs(effic_abia)/np.abs(effic_full)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower',cmap='RdBu_r',norm=MidpointNormalize(midpoint=0))
#img1=ax.imshow(np.abs(10*np.log10(np.abs(effic_abia)/np.abs(Cba**2))),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
#img1=ax.imshow(10*np.log10(np.abs(effic_abia)/np.abs(Cba**2)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
#plt.plot(p['No']*p['gm']**2/delta2vals2,delta2vals2)
plt.title(' |C|^2 (dB scale)')
plt.ylabel('atom delta')
plt.xlabel('Cavity delta')
fig.colorbar(img1)

ax=fig.add_subplot(3,1,2)
img1=ax.imshow((np.abs(effic_abia)/np.abs(effic_full)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
plt.title('|C|^2 ')
plt.ylabel('atom delta')
plt.xlabel('Cavity delta')
fig.colorbar(img1, format='%.1e')

ax=fig.add_subplot(3,1,3)
img1=ax.imshow(good_vals_abia,extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
plt.title('good?')
plt.ylabel('atom delta')
plt.xlabel('Cavity delta')
fig.colorbar(img1)

fig.suptitle('effic_adiabat/effic_full')


fig=plt.figure(filename+'effic_lin/effic_full')

fig.clf()

ax=fig.add_subplot(2,1,1)
#img1=ax.imshow(10*np.log10(np.abs(effic_abia)/np.abs(Cba**2)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower',cmap='RdBu')
img1=ax.imshow(np.log10(np.abs(Cba**2)/np.abs(effic_full)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower',cmap='RdBu_r',norm=MidpointNormalize(midpoint=0))
#img1=ax.imshow(np.abs(10*np.log10(np.abs(effic_abia)/np.abs(Cba**2))),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
#img1=ax.imshow(10*np.log10(np.abs(effic_abia)/np.abs(Cba**2)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
#plt.plot(p['No']*p['gm']**2/delta2vals2,delta2vals2)
plt.title(' |C|^2 (dB scale)')
plt.ylabel('atom delta')
plt.xlabel('Cavity delta')
fig.colorbar(img1)

ax=fig.add_subplot(2,1,2)
img1=ax.imshow((np.abs(Cba**2)/np.abs(effic_full)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
plt.title('|C|^2 ')
plt.ylabel('atom delta')
plt.xlabel('Cavity delta')
fig.colorbar(img1, format='%.1e')


fig.suptitle('effic_linear/effic_full')


fig=plt.figure(filename+'Compare thingzz log')

fig.clf()

ax=fig.add_subplot(3,1,1)
#img1=ax.imshow(10*np.log10(np.abs(effic_abia)/np.abs(Cba**2)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower',cmap='RdBu')
img1=ax.imshow(np.log10(np.abs(effic_abia)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
plt.title(' Adiabat')
plt.ylabel('atom delta')
plt.xlabel('Cavity delta')
fig.colorbar(img1)

ax=fig.add_subplot(3,1,2)
img1=ax.imshow(np.log10(np.abs(Cba**2)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
plt.title(' Linear')
plt.ylabel('atom delta')
plt.xlabel('Cavity delta')
fig.colorbar(img1)

ax=fig.add_subplot(3,1,3)
img1=ax.imshow(np.log10(np.abs(effic_full)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
plt.title('Full')
plt.ylabel('atom delta')
plt.xlabel('Cavity delta')
fig.colorbar(img1)

fig.suptitle('Comparing things -- log scale')



# fig=plt.figure(filename+'Compare thingzz lin')
#
# fig.clf()
#
# ax=fig.add_subplot(3,1,1)
# #img1=ax.imshow(10*np.log10(np.abs(effic_abia)/np.abs(Cba**2)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower',cmap='RdBu')
# img1=ax.imshow((np.abs(effic_abia)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
# plt.title(' Adiabat')
# plt.ylabel('atom delta')
# plt.xlabel('Cavity delta')
# fig.colorbar(img1)
#
# ax=fig.add_subplot(3,1,2)
# img1=ax.imshow((np.abs(Cba**2)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
# plt.title(' Linear')
# plt.ylabel('atom delta')
# plt.xlabel('Cavity delta')
# fig.colorbar(img1)
#
# ax=fig.add_subplot(3,1,3)
# img1=ax.imshow((np.abs(effic_full)),extent=(np.min(deltamuvals),np.max(deltamuvals),np.min(delta2vals),np.max(delta2vals)),aspect='auto',origin='lower')
# plt.title('Full')
# plt.ylabel('atom delta')
# plt.xlabel('Cavity delta')
# fig.colorbar(img1)
#
# fig.suptitle('Comparing things -- lin scale')


plt.show()
