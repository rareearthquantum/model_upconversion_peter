import numpy as np
import matplotlib.pyplot as plt
#plt.ion()
def find_max(matin,delovals,delmvals):
    delmmax=delmvals[np.argmax(np.abs(matin),axis=1)]

    delomax=delovals[np.argmax(np.abs(matin),axis=0)]
    return delomax,delmmax

def plot_output(filename):
    aoutpos=[2,6,1,5]
    boutpos=[4,8,3,7]
    aeffpos=[10,14,9,13]
    beffpos=[12,16,11,15]

    fig=plt.figure(filename)

    fig.clf()


    for ii in [0,1,2,3]:
        try:
            npzfile=np.load(filename+str(ii+1)+'.npz')
            aoutvals=npzfile['aoutvals']
            boutvals=npzfile['boutvals']
            effic_a=npzfile['effic_a']
            effic_b=npzfile['effic_b']
            delovals=npzfile['delovals']
            delmvals=npzfile['delmvals']
            ainvals=npzfile['ainvals']
            binvals=npzfile['binvals']
            p=npzfile['p']
            p=p.item()
            aoutmaxo,aoutmaxm=find_max(aoutvals[0,0,:,:],delovals,delmvals)
            boutmaxo,boutmaxm=find_max(boutvals[0,0,:,:],delovals,delmvals)
            effic_amaxo,effic_amaxm=find_max(effic_a[0,0,:,:],delovals,delmvals)
            effic_bmaxo,effic_bmaxm=find_max(effic_b[0,0,:,:],delovals,delmvals)

            ax=fig.add_subplot(4,4,aoutpos[ii])
            img1=ax.imshow(np.abs(aoutvals[0,0,:,:]),extent=(np.min(delmvals),np.max(delmvals),np.min(delovals),np.max(delovals)),aspect='auto',origin='lower')
            #ax.plot(delmvals,-aoutmaxo,'.-')
            #ax.plot(aoutmaxm,-delovals,'.-')
            ax.grid()
            plt.title('aout')
            plt.xlabel('delta_mu')
            plt.ylabel('delta_o')
            fig.colorbar(img1)

            ax=fig.add_subplot(4,4,boutpos[ii])
            img1=ax.imshow(np.abs(boutvals[0,0,:,:]),extent=(np.min(delmvals),np.max(delmvals),np.min(delovals),np.max(delovals)),aspect='auto',origin='lower')
            ax.grid()
            plt.title('bout')
            plt.xlabel('delta_mu')
            plt.ylabel('delta_o')
            fig.colorbar(img1)

            ax=fig.add_subplot(4,4,aeffpos[ii])
            img1=ax.imshow(np.abs(effic_a[0,0,:,:]),extent=(np.min(delmvals),np.max(delmvals),np.min(delovals),np.max(delovals)),aspect='auto',origin='lower')
            ax.grid()
            plt.title('effic_a')
            plt.xlabel('delta_mu')
            plt.ylabel('delta_o')
            fig.colorbar(img1)


            ax=fig.add_subplot(4,4,beffpos[ii])
            img1=ax.imshow(np.abs(effic_b[0,0,:,:]),extent=(np.min(delmvals),np.max(delmvals),np.min(delovals),np.max(delovals)),aspect='auto',origin='lower')
            ax.grid()
            plt.title('effic_b')
            plt.xlabel('delta_mu')
            plt.ylabel('delta_o')
            fig.colorbar(img1)
        except FileNotFoundError:
            print('Whoops, no file called ' + filename+str(ii+1)+'.npz')
    try:
        fig.suptitle('ain = ' +str(ainvals[0]) + ', bin = ' + str(binvals[0]) + ', N_bath = ' + str(p['nbath']))
    except UnboundLocalError:
        print('duh')
    #plt.show()



# plt.close('all')
# plot_output('output_ain1.npz')
# plot_output('output_ain2.npz')
# plot_output('output_bin2.npz')
# plot_output('output_bin1.npz')
# plot_output('output_binain1.npz')
# plot_output('output_binain2.npz')
#
#
# plot_output('output_ain1_2.npz')
# plot_output('output_ain2_2.npz')
# plot_output('output_bin2_2.npz')
# plot_output('output_bin1_2.npz')
for NN in [20,10,1,0]:
    for input_field in ['bin', 'ain', 'ainbin']:
            #try:
        npz_file='Runs_test1/output_N'+str(NN) +'_'+ input_field
        plot_output(npz_file)

plt.show()
