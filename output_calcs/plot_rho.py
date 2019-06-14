import numpy as np
import matplotlib.pyplot as plt
filename='output_calcs/three_lvl_testrhoc1'
filename1='output_calcs/three_lvl_out_realtest3_1'

npzfile=np.load(filename+'.npz')
rho_out=npzfile['rho_out']
npzfile1=np.load(filename1+'.npz')
rho_out1=npzfile1['rho_out']
deltamvals=npzfile['deltamvals']
deltamacvals=npzfile['deltamacvals']

fig=plt.figure(filename)
pltcount=1
fig.clf()

for ii in range(3):
    for jj in range(3):

        ax=fig.add_subplot(3,3,pltcount)
        img1=ax.imshow(np.abs(rho_out[ii,jj,:,:]-rho_out1[ii,jj,:,:]),extent=(np.min(deltamvals),np.max(deltamvals),np.min(deltamacvals),np.max(deltamacvals)),aspect='equal',origin='lower')
        plt.title('rho_' +str(ii) +str(jj)+'diff')
        plt.xlabel('delta_mu')
        plt.ylabel('deltaac_mu')
        fig.colorbar(img1)
        pltcount+=1
plt.show()
