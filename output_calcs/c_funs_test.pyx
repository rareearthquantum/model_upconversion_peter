
cimport cython
import numpy as np
#from cython.parallel cimport prange
cimport scipy.linalg.cython_lapack as lapack
#import numpy as np
#from libc.math cimport exp

def steady_rho_single_c(double delta_a_o, double delta_a_mu, double complex aval,double complex bval,double delta_o,double delta_mu,p):

    cdef double Lreal[81]
    cdef double V[9]
    cdef int workspacec[9]
    cdef int info
    cdef int lda = 9
    cdef int ldb = 9
    cdef int nrhs = 1
    cdef int n = 9
    cdef double gamma_13 = p['gamma13']
    cdef double gamma_23 = p['gamma23']
    cdef double gamma_2d = p['gamma2d']
    cdef double gamma_3d = p['gamma3d']
    cdef double n_b = p['nbath']
    cdef double gamma_mu = p['gammamu']
    cdef double g_mu = p['gm']
    cdef double g_o = p['go']
    cdef double Omega = p['Omega']
    cdef double a_r=aval.real
    cdef double a_i=aval.imag
    cdef double b_r=bval.real
    cdef double b_i=bval.imag
    V[:]=[1,0,0, 0,0,0, 0,0,0]

    Lreal[0] =1
    Lreal[1] =gamma_mu*n_b
    Lreal[2] =0
    Lreal[3] =-b_i*g_mu
    Lreal[4] =b_r*g_mu
    Lreal[5] =-a_i*g_o
    Lreal[6] =a_r*g_o
    Lreal[7] =0
    Lreal[8] =0
    Lreal[9] =1
    Lreal[10] =-gamma_mu*(n_b + 1)
    Lreal[11] =0
    Lreal[12] =b_i*g_mu
    Lreal[13] =-b_r*g_mu
    Lreal[14] =0
    Lreal[15] =0
    Lreal[16] =0
    Lreal[17] =Omega
    Lreal[18] =1
    Lreal[19] =gamma_23
    Lreal[20] =-gamma_13 - gamma_23
    Lreal[21] =0
    Lreal[22] =0
    Lreal[23] =a_i*g_o
    Lreal[24] =-a_r*g_o
    Lreal[25] =0
    Lreal[26] =-Omega
    Lreal[27] =0
    Lreal[28] =-2*b_i*g_mu
    Lreal[29] =0
    Lreal[30] =-gamma_2d/2 - gamma_mu*n_b - gamma_mu/2
    Lreal[31] =delta_a_mu - delta_mu
    Lreal[32] =0
    Lreal[33] =Omega
    Lreal[34] =-a_i*g_o
    Lreal[35] =a_r*g_o
    Lreal[36] =0
    Lreal[37] =2*b_r*g_mu
    Lreal[38] =0
    Lreal[39] =-delta_a_mu + delta_mu
    Lreal[40] =-gamma_2d/2 - gamma_mu*n_b - gamma_mu/2
    Lreal[41] =-Omega
    Lreal[42] =0
    Lreal[43] =a_r*g_o
    Lreal[44] =a_i*g_o
    Lreal[45] =0
    Lreal[46] =0
    Lreal[47] =-2*a_i*g_o
    Lreal[48] =0
    Lreal[49] =Omega
    Lreal[50] =-gamma_13/2 - gamma_23/2 - gamma_3d/2 - gamma_mu*n_b/2
    Lreal[51] =delta_a_o - delta_o
    Lreal[52] =-b_i*g_mu
    Lreal[53] =-b_r*g_mu
    Lreal[54] =0
    Lreal[55] =0
    Lreal[56] =2*a_r*g_o
    Lreal[57] =-Omega
    Lreal[58] =0
    Lreal[59] =-delta_a_o + delta_o
    Lreal[60] =-gamma_13/2 - gamma_23/2 - gamma_3d/2 - gamma_mu*n_b/2
    Lreal[61] =b_r*g_mu
    Lreal[62] =-b_i*g_mu
    Lreal[63] =0
    Lreal[64] =0
    Lreal[65] =0
    Lreal[66] =a_i*g_o
    Lreal[67] =-a_r*g_o
    Lreal[68] =b_i*g_mu
    Lreal[69] =-b_r*g_mu
    Lreal[70] =-gamma_13/2 - gamma_23/2 - gamma_2d/2 - gamma_3d/2 - gamma_mu*(n_b + 1)/2
    Lreal[71] =-delta_a_mu + delta_a_o + delta_mu - delta_o
    Lreal[72] =0
    Lreal[73] =-2*Omega
    Lreal[74] =2*Omega
    Lreal[75] =-a_r*g_o
    Lreal[76] =-a_i*g_o
    Lreal[77] =b_r*g_mu
    Lreal[78] =b_i*g_mu
    Lreal[79] =delta_a_mu - delta_a_o - delta_mu + delta_o
    Lreal[80] =-gamma_13/2 - gamma_23/2 - gamma_2d/2 - gamma_3d/2 - gamma_mu*(n_b + 1)/2

    #rho = np.linalg.solve(L_mat.astype('complex128'),np.matrix([[1,0,0,0,0,0,0,0,0]]).T)
    lapack.dgesv(&n,&nrhs,Lreal,&lda,workspacec,V,&ldb, &info)
    #lapack.dgesv(&n,&nrhs,L,&lda,workspace,V,&ldb,&info)
    #return V
    #return [V[3]+1j*V[4],V[5]+1j*V[6]]
    return np.array([[V[0],V[3]+1j*V[4],V[5]+1j*V[6]],[V[3]-1j*V[4],V[1],V[7]+1j*V[8]],[V[5]-1j*V[6],V[7]-1j*V[8],V[2]]])
