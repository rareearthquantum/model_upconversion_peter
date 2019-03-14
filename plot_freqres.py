import numpy as np
import matplotlib.pyplot as plt
#plt.ion()
def find_max(matin,delovals,delmvals):
    delmmax=delmvals[np.argmax(np.abs(matin),axis=1)]

    delomax=delovals[np.argmax(np.abs(matin),axis=0)]
    return delomax,delmmax

def plot_output(filename):
    npzfile=np.load(filename)
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

    fig=plt.figure(filename)

    fig.clf()
    fig.suptitle('ain = ' +str(ainvals[0]) + ', bin = ' + str(binvals[0]) + ', N_bath = ' + str(p['nbath']))
    ax=fig.add_subplot(2,2,1)
    img1=ax.imshow(np.abs(aoutvals[0,0,:,:]),extent=(np.min(delmvals),np.max(delmvals),np.min(delovals),np.max(delovals)),aspect='auto')
    #ax.plot(delmvals,-aoutmaxo,'.-')
    #ax.plot(aoutmaxm,-delovals,'.-')
    ax.grid()
    plt.title('aout')
    plt.xlabel('delta_mu')
    plt.ylabel('delta_o')
    fig.colorbar(img1)

    ax=fig.add_subplot(2,2,2)
    img1=ax.imshow(np.abs(boutvals[0,0,:,:]),extent=(np.min(delmvals),np.max(delmvals),np.min(delovals),np.max(delovals)),aspect='auto')
    ax.grid()
    plt.title('bout')
    plt.xlabel('delta_mu')
    plt.ylabel('delta_o')
    fig.colorbar(img1)

    ax=fig.add_subplot(2,2,3)
    img1=ax.imshow(np.abs(effic_a[0,0,:,:]),extent=(np.min(delmvals),np.max(delmvals),np.min(delovals),np.max(delovals)),aspect='auto')
    ax.grid()
    plt.title('effic_a')
    plt.xlabel('delta_mu')
    plt.ylabel('delta_o')
    fig.colorbar(img1)


    ax=fig.add_subplot(2,2,4)
    img1=ax.imshow(np.abs(effic_b[0,0,:,:]),extent=(np.min(delmvals),np.max(delmvals),np.min(delovals),np.max(delovals)),aspect='auto')
    ax.grid()
    plt.title('effic_b')
    plt.xlabel('delta_mu')
    plt.ylabel('delta_o')
    fig.colorbar(img1)

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
for NN in [20,1,0]:
    for input_field in ['ain','bin']:#, 'ain', 'ainbin']:
        for ii in [4]:
            try:
                npz_file='runs_detune_test1/output_N'+str(NN) +'_d_'+ input_field+str(ii)+'.npz'
                plot_output(npz_file)
            except FileNotFoundError:
                print('Whoops, no file called ' + npz_file)
#plt.show()
#npz_file='Runs_test1/output_N'+str(20) +'_'+ 'ain'+str(1)+'.npz'
#plot_output(npz_file)
plt.show()
