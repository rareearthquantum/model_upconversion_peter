import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
print(scipy.__version__)
import scipy.io
foldername='data4gav/'# 'Paper_figs1/'
foldername='Paper_figs1/'

filenames= ['sim_fig4_data_ds2','sim_fig6_data_dB3_test_ds2_biggish7_2', 'sim_fig5_data_temp9', 'sim_fig7_data_temp8_cfe2']
for filename in filenames:
    filename=foldername+filename
    npzfile=np.load(filename+'.npz')
    print(list(npzfile.keys()))
    scipy.io.savemat(filename +'.mat', mdict={key:npzfile[key] for key in npzfile.keys()})
