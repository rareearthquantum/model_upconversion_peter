
import numpy as np
import matplotlib.pyplot as plt


def combine_data_plot(filestart,wantsave):
    npz1=np.load(filestart + '1'+'.npz')
    npz2=np.load(filestart + '2'+'.npz')
    npz3=np.load(filestart + '3'+'.npz')
    npz4=np.load(filestart + '4'+'.npz')
    bo1=npz1['boutvals']
    bo2=npz2['boutvals']
    bo3=npz3['boutvals']
    bo4=npz4['boutvals']


    ao1=npz1['aoutvals']
    ao2=npz2['aoutvals']
    ao3=npz3['aoutvals']
    ao4=npz4['aoutvals']

    ae1=npz1['effic_a']
    ae2=npz2['effic_a']
    ae3=npz3['effic_a']
    ae4=npz4['effic_a']

    be1=npz1['effic_b']
    be2=npz2['effic_b']
    be3=npz3['effic_b']
    be4=npz4['effic_b']

    #boutvals=[[bo3[0,0,:,:],bo1[0,0,:,:]],[bo4[0,0,:,:],bo2[0,0,:,:]]]
    boutvals=np.hstack((np.vstack((bo4[0,0,:,:],bo3[0,0,1:,:])),np.vstack((bo2[0,0,:,1:],bo1[0,0,1:,1:]))))
    aoutvals=np.hstack((np.vstack((ao4[0,0,:,:],ao3[0,0,1:,:])),np.vstack((ao2[0,0,:,1:],ao1[0,0,1:,1:]))))

    effic_b=np.hstack((np.vstack((be4[0,0,:,:],be3[0,0,1:,:])),np.vstack((be2[0,0,:,1:],be1[0,0,1:,1:]))))
    effic_a=np.hstack((np.vstack((ae4[0,0,:,:],ae3[0,0,1:,:])),np.vstack((ae2[0,0,:,1:],ae1[0,0,1:,1:]))))
    fig=plt.figure(filestart)
    ainvals=npz1['ainvals']
    binvals=npz1['binvals']
    p=npz1['p']
    p=p.item()
    delovals=np.concatenate((npz4['delovals'], npz3['delovals'][1:]))
    delmvals=np.concatenate((npz4['delovals'], npz2['delovals'][1:]))
    fig.clf()
    fig.suptitle('ain = ' +str(ainvals[0]) + ', bin = ' + str(binvals[0]) + ', N_bath = ' + str(p['nbath']))
    ax=fig.add_subplot(2,2,1)
    img1=ax.imshow(np.abs(aoutvals),extent=(np.min(delmvals),np.max(delmvals),np.min(delovals),np.max(delovals)),aspect='auto',origin='lower')
    #ax.plot(delmvals,-aoutmaxo,'.-')
    #ax.plot(aoutmaxm,-delovals,'.-')

    ax.grid()
    plt.title('aout')
    plt.xlabel('delta_mu')
    plt.ylabel('delta_o')
    fig.colorbar(img1)

    ax=fig.add_subplot(2,2,2)
    img1=ax.imshow(np.abs(boutvals),extent=(np.min(delmvals),np.max(delmvals),np.min(delovals),np.max(delovals)),aspect='auto',origin='lower')
    ax.grid()
    plt.title('bout')
    plt.xlabel('delta_mu')
    plt.ylabel('delta_o')
    fig.colorbar(img1)

    ax=fig.add_subplot(2,2,3)
    img1=ax.imshow(np.log10(np.abs(effic_a)),extent=(np.min(delmvals),np.max(delmvals),np.min(delovals),np.max(delovals)),aspect='auto',origin='lower')
    ax.grid()
    plt.title('effic_a')
    plt.xlabel('delta_mu')
    plt.ylabel('delta_o')
    fig.colorbar(img1)


    ax=fig.add_subplot(2,2,4)
    img1=ax.imshow(np.log10(np.abs(effic_b)),extent=(np.min(delmvals),np.max(delmvals),np.min(delovals),np.max(delovals)),aspect='auto',origin='lower')
    ax.grid()
    plt.title('effic_b')
    plt.xlabel('delta_mu')
    plt.ylabel('delta_o')
    fig.colorbar(img1)
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    if wantsave:
        plt.savefig(filestart + 'fig.pdf')


for NN in [20,1,0]:
    for input_field in ['bin', 'ain', 'ainbin']:
            try:
                npz_file='runs_detune_test1/output_N'+str(NN) +'_d_'+ input_field
                combine_data_plot(npz_file,0)
            except FileNotFoundError:
                print('no file lol ')

plt.show()
