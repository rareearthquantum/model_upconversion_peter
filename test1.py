import numpy as np
#for ii in [23,27,28,29,30,31]:

filename='Paper_figs1/sim_fig7_data_test2'
npzfile=np.load(filename+'.npz')
p1=npzfile['p'][()]
#print(ii)
print(p1['mean_delao'])
print(npzfile['P_mu'])
print(p1)

# filename='Paper_figs1/sim_fig7_data_test3'
# npzfile=np.load(filename+'.npz')
# p2=npzfile['p'][()]
# print(npzfile['P_mu'])
# print(p2)
#
# filename='Paper_figs1/sim_fig7_data_temp6'
# npzfile=np.load(filename+'.npz')
# p3=npzfile['p'][()]
# print(npzfile['P_mu'])
# print(p3)
