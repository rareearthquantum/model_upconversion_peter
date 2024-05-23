
cimport cython
#cimport numpy as np
import numpy as np

from cython.parallel cimport prange
cimport scipy.linalg.cython_lapack as lapack
#from scipy.linalg.cython_blas cimport dgemv

#import numpy as np
from libc.math cimport exp, sqrt,abs, copysign
import scipy.special#.roots_legendre
#cdef double complex steady_rhoc1[3](double a_r):
#    cdef int aa = 2
#    cdef double bb[3]
#    bb[2]=a_r*aa
#    return bb
#hh=steady_rhoc1(1.9)
#print('  ' +str(hh))
#def Lfuncflat(aval, bval,deloval,delmval, delaoval,delamval,p):
#    return Lfunc(aval, bval,deloval,delmval, delaoval,delamval,p['gamma13'],p['gamma23'],p['gamma2d'],p['gamma3d'], p['nbath'],p['gammamu'],p['Omega'],p['go'],p['gm']).T.flatten()
# cdef extern from 'cblas.h':
#     ctypedef enum CBLAS_ORDER:
#         CblasRowMajor
#         CblasColMajor
#     ctypedef enum CBLAS_TRANSPOSE:
#         CblasNoTrans
#         CblasTrans
#         CblasConjTrans
#     void dgemv 'cblas_dgemv'(CBLAS_ORDER order,
#                              CBLAS_TRANSPOSE transpose,
#                              int M, int N,
#                              double alpha, double* A, int lda,
#                              double* X, int incX,
#                              double beta, double* Y, int incY) nogil
# cpdef run_blas_dgemv(double[:, ::1] A,
#                      double[::1] x,
#                      double[::1] y,
#                      int M,
#                      int N,
#                      double alpha,
#                      double beta):
#
#     cdef double* A_ptr = &A[0, 0]
#     cdef double* x_ptr = &x[0]
#     cdef double* y_ptr = &y[0]
#
#     #dgemv(CblasRowMajor,
#     dgemv('N',#CblasNoTrans,
#           M,
#           N,
#           alpha,
#           A_ptr,
#           N,
#           x_ptr,
#           1,
#           beta,
#           y_ptr,
#           1)
cdef Lx_rho0(double[::1] Lx,double[::1] rho0):
    cdef double Lrho0[9]
    cdef int ii
    for ii in range(9):
        #Lrho0[ii]=rho0[0]*Lx[9*ii]+rho0[1]*Lx[9*ii+1]+rho0[2]*Lx[9*ii+2]+rho0[7]*Lx[9*ii+7]+rho0[8]*Lx[9*ii+8]
        Lrho0[ii]=rho0[0]*Lx[ii]+rho0[1]*Lx[ii+9*1]+rho0[2]*Lx[ii+9*2]+rho0[7]*Lx[ii+9*7]+rho0[8]*Lx[ii+9*8]
    return Lrho0
#
# cdef Lx_rho0(Lx,rho0):
  # return np.matmul(np.reshape(Lx,(9,9),order='F'),np.reshape(rho0,(9,1)))
def steady_rhos(double delta_a_o, double delta_a_mu,double delta_o,double delta_mu,p):

    cdef double L0[81]
    cdef double L0_save[81]
    cdef double Lar[81]
    cdef double Lai[81]
    cdef double Lbr[81]
    cdef double Lbi[81]

    cdef double rhovec[21]
    cdef int ii = 0
    cdef int jj
    cdef int rho0inds[5] #= [0,1,2,7,8]
    cdef int rhoxinds[4] #= [3,4,5,6]
    rho0inds[:] = [0,1,2,7,8]
    rhoxinds[:] = [3,4,5,6]
    cdef double V[9]
    cdef double rho0[9]
    #cdef double rho1[9]

    #cdef double rho0
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
    #cdef
    #Lar=([[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 2*g_o, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, -g_o], [0, 0, 0, 0, 0, 0, 0, -g_o, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [g_o, 0, -g_o, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, g_o, 0, 0, 0, 0], [0, 0, 0, g_o, 0, 0, 0, 0, 0]])
    #Lai=([[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 2*g_o, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, -g_o, 0], [0, 0, 0, 0, 0, 0, 0, 0, g_o], [g_o, 0, -g_o, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, g_o, 0, 0, 0, 0, 0], [0, 0, 0, 0, -g_o, 0, 0, 0, 0]])
    #Lbr=([[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 2*g_mu, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [g_mu, -g_mu, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, g_mu], [0, 0, 0, 0, 0, 0, 0, -g_mu, 0], [0, 0, 0, 0, 0, 0, g_mu, 0, 0], [0, 0, 0, 0, 0, -g_mu, 0, 0, 0]])
    #Lbi=([[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 2*g_mu, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [g_mu, -g_mu, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, -g_mu, 0], [0, 0, 0, 0, 0, 0, 0, 0, -g_mu], [0, 0, 0, 0, 0, g_mu, 0, 0, 0], [0, 0, 0, 0, 0, 0, g_mu, 0, 0]])
    L0[0] = 1
    L0[1] = gamma_mu*n_b
    L0[2] = 0
    L0[3] = 0
    L0[4] = 0
    L0[5] = 0
    L0[6] = 0
    L0[7] = 0
    L0[8] = 0
    L0[9] = 1
    L0[10] = -gamma_mu*(n_b + 1)
    L0[11] = 0
    L0[12] = 0
    L0[13] = 0
    L0[14] = 0
    L0[15] = 0
    L0[16] = 0
    L0[17] = Omega
    L0[18] = 1
    L0[19] = gamma_23
    L0[20] = -gamma_13 - gamma_23
    L0[21] = 0
    L0[22] = 0
    L0[23] = 0
    L0[24] = 0
    L0[25] = 0
    L0[26] = -Omega
    L0[27] = 0
    L0[28] = 0
    L0[29] = 0
    L0[30] = -gamma_2d/2 - gamma_mu*n_b - gamma_mu/2
    L0[31] = delta_a_mu - delta_mu
    L0[32] = 0
    L0[33] = Omega
    L0[34] = 0
    L0[35] = 0
    L0[36] = 0
    L0[37] = 0
    L0[38] = 0
    L0[39] = -delta_a_mu + delta_mu
    L0[40] = -gamma_2d/2 - gamma_mu*n_b - gamma_mu/2
    L0[41] = -Omega
    L0[42] = 0
    L0[43] = 0
    L0[44] = 0
    L0[45] = 0
    L0[46] = 0
    L0[47] = 0
    L0[48] = 0
    L0[49] = Omega
    L0[50] = -gamma_13/2 - gamma_23/2 - gamma_3d/2 - gamma_mu*n_b/2
    L0[51] = delta_a_o - delta_o
    L0[52] = 0
    L0[53] = 0
    L0[54] = 0
    L0[55] = 0
    L0[56] = 0
    L0[57] = -Omega
    L0[58] = 0
    L0[59] = -delta_a_o + delta_o
    L0[60] = -gamma_13/2 - gamma_23/2 - gamma_3d/2 - gamma_mu*n_b/2
    L0[61] = 0
    L0[62] = 0
    L0[63] = 0
    L0[64] = 0
    L0[65] = 0
    L0[66] = 0
    L0[67] = 0
    L0[68] = 0
    L0[69] = 0
    L0[70] = -gamma_13/2 - gamma_23/2 - gamma_2d/2 - gamma_3d/2 - gamma_mu*n_b/2 - gamma_mu/2
    L0[71] = -delta_a_mu + delta_a_o + delta_mu - delta_o
    L0[72] = 0
    L0[73] = -2*Omega
    L0[74] = 2*Omega
    L0[75] = 0
    L0[76] = 0
    L0[77] = 0
    L0[78] = 0
    L0[79] = delta_a_mu - delta_a_o - delta_mu + delta_o
    L0[80] = -gamma_13/2 - gamma_23/2 - gamma_2d/2 - gamma_3d/2 - gamma_mu*n_b/2 - gamma_mu/2
    Lar[0] = 0
    Lar[1] = 0
    Lar[2] = 0
    Lar[3] = 0
    Lar[4] = 0
    Lar[5] = 0
    Lar[6] = g_o
    Lar[7] = 0
    Lar[8] = 0
    Lar[9] = 0
    Lar[10] = 0
    Lar[11] = 0
    Lar[12] = 0
    Lar[13] = 0
    Lar[14] = 0
    Lar[15] = 0
    Lar[16] = 0
    Lar[17] = 0
    Lar[18] = 0
    Lar[19] = 0
    Lar[20] = 0
    Lar[21] = 0
    Lar[22] = 0
    Lar[23] = 0
    Lar[24] = -g_o
    Lar[25] = 0
    Lar[26] = 0
    Lar[27] = 0
    Lar[28] = 0
    Lar[29] = 0
    Lar[30] = 0
    Lar[31] = 0
    Lar[32] = 0
    Lar[33] = 0
    Lar[34] = 0
    Lar[35] = g_o
    Lar[36] = 0
    Lar[37] = 0
    Lar[38] = 0
    Lar[39] = 0
    Lar[40] = 0
    Lar[41] = 0
    Lar[42] = 0
    Lar[43] = g_o
    Lar[44] = 0
    Lar[45] = 0
    Lar[46] = 0
    Lar[47] = 0
    Lar[48] = 0
    Lar[49] = 0
    Lar[50] = 0
    Lar[51] = 0
    Lar[52] = 0
    Lar[53] = 0
    Lar[54] = 0
    Lar[55] = 0
    Lar[56] = 2*g_o
    Lar[57] = 0
    Lar[58] = 0
    Lar[59] = 0
    Lar[60] = 0
    Lar[61] = 0
    Lar[62] = 0
    Lar[63] = 0
    Lar[64] = 0
    Lar[65] = 0
    Lar[66] = 0
    Lar[67] = -g_o
    Lar[68] = 0
    Lar[69] = 0
    Lar[70] = 0
    Lar[71] = 0
    Lar[72] = 0
    Lar[73] = 0
    Lar[74] = 0
    Lar[75] = -g_o
    Lar[76] = 0
    Lar[77] = 0
    Lar[78] = 0
    Lar[79] = 0
    Lar[80] = 0
    Lai[0] = 0
    Lai[1] = 0
    Lai[2] = 0
    Lai[3] = 0
    Lai[4] = 0
    Lai[5] = g_o
    Lai[6] = 0
    Lai[7] = 0
    Lai[8] = 0
    Lai[9] = 0
    Lai[10] = 0
    Lai[11] = 0
    Lai[12] = 0
    Lai[13] = 0
    Lai[14] = 0
    Lai[15] = 0
    Lai[16] = 0
    Lai[17] = 0
    Lai[18] = 0
    Lai[19] = 0
    Lai[20] = 0
    Lai[21] = 0
    Lai[22] = 0
    Lai[23] = -g_o
    Lai[24] = 0
    Lai[25] = 0
    Lai[26] = 0
    Lai[27] = 0
    Lai[28] = 0
    Lai[29] = 0
    Lai[30] = 0
    Lai[31] = 0
    Lai[32] = 0
    Lai[33] = 0
    Lai[34] = g_o
    Lai[35] = 0
    Lai[36] = 0
    Lai[37] = 0
    Lai[38] = 0
    Lai[39] = 0
    Lai[40] = 0
    Lai[41] = 0
    Lai[42] = 0
    Lai[43] = 0
    Lai[44] = -g_o
    Lai[45] = 0
    Lai[46] = 0
    Lai[47] = 2*g_o
    Lai[48] = 0
    Lai[49] = 0
    Lai[50] = 0
    Lai[51] = 0
    Lai[52] = 0
    Lai[53] = 0
    Lai[54] = 0
    Lai[55] = 0
    Lai[56] = 0
    Lai[57] = 0
    Lai[58] = 0
    Lai[59] = 0
    Lai[60] = 0
    Lai[61] = 0
    Lai[62] = 0
    Lai[63] = 0
    Lai[64] = 0
    Lai[65] = 0
    Lai[66] = -g_o
    Lai[67] = 0
    Lai[68] = 0
    Lai[69] = 0
    Lai[70] = 0
    Lai[71] = 0
    Lai[72] = 0
    Lai[73] = 0
    Lai[74] = 0
    Lai[75] = 0
    Lai[76] = g_o
    Lai[77] = 0
    Lai[78] = 0
    Lai[79] = 0
    Lai[80] = 0
    Lbr[0] = 0
    Lbr[1] = 0
    Lbr[2] = 0
    Lbr[3] = 0
    Lbr[4] = g_mu
    Lbr[5] = 0
    Lbr[6] = 0
    Lbr[7] = 0
    Lbr[8] = 0
    Lbr[9] = 0
    Lbr[10] = 0
    Lbr[11] = 0
    Lbr[12] = 0
    Lbr[13] = -g_mu
    Lbr[14] = 0
    Lbr[15] = 0
    Lbr[16] = 0
    Lbr[17] = 0
    Lbr[18] = 0
    Lbr[19] = 0
    Lbr[20] = 0
    Lbr[21] = 0
    Lbr[22] = 0
    Lbr[23] = 0
    Lbr[24] = 0
    Lbr[25] = 0
    Lbr[26] = 0
    Lbr[27] = 0
    Lbr[28] = 0
    Lbr[29] = 0
    Lbr[30] = 0
    Lbr[31] = 0
    Lbr[32] = 0
    Lbr[33] = 0
    Lbr[34] = 0
    Lbr[35] = 0
    Lbr[36] = 0
    Lbr[37] = 2*g_mu
    Lbr[38] = 0
    Lbr[39] = 0
    Lbr[40] = 0
    Lbr[41] = 0
    Lbr[42] = 0
    Lbr[43] = 0
    Lbr[44] = 0
    Lbr[45] = 0
    Lbr[46] = 0
    Lbr[47] = 0
    Lbr[48] = 0
    Lbr[49] = 0
    Lbr[50] = 0
    Lbr[51] = 0
    Lbr[52] = 0
    Lbr[53] = -g_mu
    Lbr[54] = 0
    Lbr[55] = 0
    Lbr[56] = 0
    Lbr[57] = 0
    Lbr[58] = 0
    Lbr[59] = 0
    Lbr[60] = 0
    Lbr[61] = g_mu
    Lbr[62] = 0
    Lbr[63] = 0
    Lbr[64] = 0
    Lbr[65] = 0
    Lbr[66] = 0
    Lbr[67] = 0
    Lbr[68] = 0
    Lbr[69] = -g_mu
    Lbr[70] = 0
    Lbr[71] = 0
    Lbr[72] = 0
    Lbr[73] = 0
    Lbr[74] = 0
    Lbr[75] = 0
    Lbr[76] = 0
    Lbr[77] = g_mu
    Lbr[78] = 0
    Lbr[79] = 0
    Lbr[80] = 0
    Lbi[0] = 0
    Lbi[1] = 0
    Lbi[2] = 0
    Lbi[3] = g_mu
    Lbi[4] = 0
    Lbi[5] = 0
    Lbi[6] = 0
    Lbi[7] = 0
    Lbi[8] = 0
    Lbi[9] = 0
    Lbi[10] = 0
    Lbi[11] = 0
    Lbi[12] = -g_mu
    Lbi[13] = 0
    Lbi[14] = 0
    Lbi[15] = 0
    Lbi[16] = 0
    Lbi[17] = 0
    Lbi[18] = 0
    Lbi[19] = 0
    Lbi[20] = 0
    Lbi[21] = 0
    Lbi[22] = 0
    Lbi[23] = 0
    Lbi[24] = 0
    Lbi[25] = 0
    Lbi[26] = 0
    Lbi[27] = 0
    Lbi[28] = 2*g_mu
    Lbi[29] = 0
    Lbi[30] = 0
    Lbi[31] = 0
    Lbi[32] = 0
    Lbi[33] = 0
    Lbi[34] = 0
    Lbi[35] = 0
    Lbi[36] = 0
    Lbi[37] = 0
    Lbi[38] = 0
    Lbi[39] = 0
    Lbi[40] = 0
    Lbi[41] = 0
    Lbi[42] = 0
    Lbi[43] = 0
    Lbi[44] = 0
    Lbi[45] = 0
    Lbi[46] = 0
    Lbi[47] = 0
    Lbi[48] = 0
    Lbi[49] = 0
    Lbi[50] = 0
    Lbi[51] = 0
    Lbi[52] = g_mu
    Lbi[53] = 0
    Lbi[54] = 0
    Lbi[55] = 0
    Lbi[56] = 0
    Lbi[57] = 0
    Lbi[58] = 0
    Lbi[59] = 0
    Lbi[60] = 0
    Lbi[61] = 0
    Lbi[62] = g_mu
    Lbi[63] = 0
    Lbi[64] = 0
    Lbi[65] = 0
    Lbi[66] = 0
    Lbi[67] = 0
    Lbi[68] = -g_mu
    Lbi[69] = 0
    Lbi[70] = 0
    Lbi[71] = 0
    Lbi[72] = 0
    Lbi[73] = 0
    Lbi[74] = 0
    Lbi[75] = 0
    Lbi[76] = 0
    Lbi[77] = 0
    Lbi[78] = -g_mu
    Lbi[79] = 0
    Lbi[80] = 0
    L0_save=L0
    #rhovec=np.zeros(45)
    rho0[:]=[1,0,0, 0,0,0, 0,0,0]
    #for ii in range(9):
    #  rhovec[
    #rho = np.linalg.solve(L_mat.astype('complex128'),np.matrix([[1,0,0,0,0,0,0,0,0]]).T)
    #L0_mat=np.array(np.reshape(np.array(L0),(9,9),order='F'))

    #Lar_mat=np.array(np.reshape(np.array(Lar),(9,9),order='F'))
    #Lai_mat=np.array(np.reshape(np.array(Lai),(9,9),order='F'))
    #Lbr_mat=np.array(np.reshape(np.array(Lbr),(9,9),order='F'))
    #Lbi_mat=np.array(np.reshape(np.array(Lbi),(9,9),order='F'))
    #L0_inv=np.linalg.inv(L0_mat)

    lapack.dgesv(&n,&nrhs,L0,&lda,workspacec,rho0,&ldb, &info)
    #L0=L0_save
    for jj in range(5):
      rhovec[jj]=rho0[rho0inds[jj]]
    #for Lx in [Lar, Lai, Lbr, Lbi]:
    for ii in range(4):
      #V[:]=rho0
      # if ii == 0:
      #   dgemv('N',&m,&n,&alpha,Lar,&lda, V,&incx,&beta,Y,&incy)
      # elif ii == 1:
      #   dgemv('N',&m,&n,&alpha,Lai,&lda, V,&incx,&beta,Y,&incy)
      # elif ii == 2:
      #   dgemv('N',&m,&n,&alpha,Lbr,&lda, V,&incx,&beta,Y,&incy)
      # elif ii == 3:
      #   dgemv('N',&m,&n,&alpha,Lbi,&lda, V,&incx,&beta,Y,&incy)
      if ii == 0:
        V[:]=Lx_rho0(Lar,rho0)
      elif ii == 1:
        V[:]=Lx_rho0(Lai,rho0)
      elif ii == 2:
        V[:]=Lx_rho0(Lbr,rho0)
      elif ii == 3:
        V[:]=Lx_rho0(Lbi,rho0)
      #rho0[:]=V
      #print(V)
      #rho1[:]=V[:]
      #lapack.dgesv(&n2,&nrhs2,L0,&lda2,workspacec2,rho1,&ldb2, &info2)
      L0=L0_save

      lapack.dgesv(&n,&nrhs,L0,&lda,workspacec,V,&ldb, &info)

      #print(np.array(V))
#      V[:]=np.linalg.lstsq(L0_mat,np.array(V).T,rcond=None)[0]
      #V[:]=np.linalg.solve(L0_mat,np.array(V).T)

      #print(V)
      #lapack.dgesv(&n,&nrhs,L0,&lda,workspacec,V,&ldb, &info)
      #print(rho0)

      for jj in range(4):
        rhovec[5+jj+4*ii]=V[rhoxinds[jj]]

    return rhovec
    # return rhovec
    #return [V[3]+1j*V[4],V[5]+1j*V[6]]
    #return np.array([[V[0],V[3]+1j*V[4],V[5]+1j*V[6]],[V[3]-1j*V[4],V[1],V[7]+1j*V[8]],[V[5]-1j*V[6],V[7]-1j*V[8],V[2]]])

def gauss_fun_1d(double x,double m, double sd):
  return 1.0/(sqrt(2*3.14159265358979323846)*sd)*exp(-(x-m)**2/(2*sd**2)) #2.5066282746310002 is sqrt(2*pi)


DEF n_gauss=25#35
DEF n_lag =25#35#15
#global xgauss, wgauss
cdef double xgauss[n_gauss]
cdef double wgauss[n_gauss]
xgauss[:],wgauss[:]=scipy.special.roots_legendre(n_gauss)
cdef double xlag[n_lag]
cdef double wlag[n_lag]
xlag[:],wlag[:]=scipy.special.roots_laguerre(n_lag)


def rho_m_broad_single(double delaoval, double deloval,double delmval, p,delamlims):
    cdef double delamlimsc[2]
    delamlimsc[:]=delamlims
    #xvals, wvals = _cached_roots_legendre(n)
    cdef double yval#s[n_gauss]
    cdef double S_out[21]
    S_out[:]=np.zeros(21)#[0,0,0, 0,0,0 ,0,0,0]
    cdef double rho_temp[21]
    cdef double gauss_weight
    cdef int ii
    cdef int jj
    cdef double mean_delam=p['mean_delam']
    cdef double sd_delam=p['sd_delam']
    #[yval[ii] = (delamlims[1]-delamlims[0])*(xgauss[ii]+1)/2.0 + delamlims[0] for ii in range(n_gauss)]

    #for ii in prange(n_gauss,nogil=True):
    for ii in range(n_gauss):
        yval=(delamlimsc[1]-delamlimsc[0])*(xgauss[ii]+1)/2.0 + delamlimsc[0]
        #gauss_weight=gauss_fun_1d(yval,p['mean_delam'],p['sd_delam'])*wgauss[ii]
        gauss_weight=gauss_fun_1d(yval,mean_delam,sd_delam)*wgauss[ii]
        rho_temp=(steady_rhos(delaoval,yval,deloval,delmval,p))
        #S_out=[S_out[jj]+rho_temp[jj]*(delamlims[1]-delamlims[0])/2.0*gauss_weight for jj in range(9)]
        for jj in range(21):
            S_out[jj]+=rho_temp[jj]*(delamlimsc[1]-delamlimsc[0])/2.0*gauss_weight
    return S_out

#def rho_m_broad_full_bad(delaoval,aval,bval,deloval,delmval, p):
#    delamlims=[-50*p['sd_delam']+p['mean_delam'],50*p['sd_delam']+p['mean_delam']]
#    #splitpoints=[p['mean_delam'],delmval,-11*p['sd_delam']+p['mean_delam'],11*p['sd_delam']+p['mean_delam']]
#    ds_m=np.nan#find_dressed_states_m(delaoval, deloval,delmval,bval,p)[0].real
#    #ds_test=(steady_rho_single(delaoval,ds_m,aval, bval,deloval,delmval,p))
#    splitpoints=[p['mean_delam'],p['sd_delam']+p['mean_delam'] ,-p['sd_delam']+p['mean_delam'],delmval,delmval+2*p['gamma2d'],delmval-2*p['gamma2d']]#,p['sd_delam']+p['mean_delam']]
#    if not np.isnan(ds_m) and not ds_m in splitpoints:
#        splitpoints.append(ds_m)
#    points=np.array(sorted(np.hstack((delamlims[0],(splitpoints),delamlims[1])))
#    #print(ds_m)
#    #print(type(ds_m))
#    S_out_full=np.zeros((3,3),dtype=np.complex)
#    for ii in range(len(points)-2):
#       S_out_full=S_out_full+rho_m_broad_single(delaoval,aval,bval,deloval,delmval, p,[points[ii], points[ii+1]],n=n)
#    return S_out_full

def rho_m_broad_single_highbound(double delaoval, double deloval,double delmval, p,double delamlim):
    #xvals, wvals = _cached_roots_legendre(n)
    #cdef double yval#s[n_gauss]
    cdef double S_out[21]
    S_out[:]=np.zeros(21)#[0,0,0, 0,0,0 ,0,0,0]
    cdef double rho_temp[21]
    cdef double gaussexp_weight
    cdef int ii
    cdef int jj
    cdef double mean_delam=p['mean_delam']
    cdef double sd_delam=p['sd_delam']
    #[yval[ii] = (delamlims[1]-delamlims[0])*(xgauss[ii]+1)/2.0 + delamlims[0] for ii in range(n_gauss)]

    #for ii in prange(n_gauss,nogil=True):
    for ii in range(n_lag):
        #yval=xlag[ii]-delamlim
        #gauss_weight=gauss_fun_1d(yval,p['mean_delam'],p['sd_delam'])*wgauss[ii]
        gaussexp_weight=gauss_fun_1d(xlag[ii]+delamlim,mean_delam,sd_delam)*wlag[ii]*exp(xlag[ii])
        rho_temp=(steady_rhos(delaoval,xlag[ii]+delamlim,deloval,delmval,p))
        #S_out=[S_out[jj]+rho_temp[jj]*(delamlims[1]-delamlims[0])/2.0*gauss_weight for jj in range(9)]
        for jj in range(21):
            S_out[jj]+=rho_temp[jj]*gaussexp_weight
    #S_out[:]=[0,0,0, 0,0,0 ,0,0,0]
    return S_out
def rho_m_broad_single_lowbound(double delaoval, double deloval,double delmval, p,double delamlim):
  #xvals, wvals = _cached_roots_legendre(n)
  #cdef double yval#s[n_gauss]
  cdef double S_out[21]
  S_out[:]=np.zeros(21)#[0,0,0, 0,0,0 ,0,0,0]
  cdef double rho_temp[21]
  cdef double gaussexp_weight
  cdef int ii
  cdef int jj
  cdef double mean_delam=p['mean_delam']
  cdef double sd_delam=p['sd_delam']
  #[yval[ii] = (delamlims[1]-delamlims[0])*(xgauss[ii]+1)/2.0 + delamlims[0] for ii in range(n_gauss)]

  #for ii in prange(n_gauss,nogil=True):
  for ii in range(n_lag):
      #yval=xlag[ii]-delamlim
      #gauss_weight=gauss_fun_1d(yval,p['mean_delam'],p['sd_delam'])*wgauss[ii]
      gaussexp_weight=gauss_fun_1d(-xlag[ii]+delamlim,mean_delam,sd_delam)*wlag[ii]*exp(xlag[ii])
      rho_temp=(steady_rhos(delaoval,-xlag[ii]+delamlim,deloval,delmval,p))
      #S_out=[S_out[jj]+rho_temp[jj]*(delamlims[1]-delamlims[0])/2.0*gauss_weight for jj in range(9)]
      for jj in range(21):
          S_out[jj]+=rho_temp[jj]*gaussexp_weight
  #S_out[:]=[0,0,0, 0,0,0 ,0,0,0]
  return S_out
def rho_m_broad_full(double delaoval,double deloval,double delmval,ds_m_fun, p):
    #cdef double points[17]#[6]
    cdef double points[32]#[6]

    cdef double S_out[21]
    cdef double S_temp[21]
    cdef double pointpairs[2]
    cdef double ds_m_val
    cdef int ii,jj
    #cdef double omegamu = p['gm']*abs(bval)
    #cdef double omegamu2 = omegamu**2
    #cdef double deltaoval = delaoval-deloval
    #if abs(deltaoval) <omegamu:
    #   deltaoval=copysign(omegamu,deltaoval)
    S_out=np.zeros(21)#[0,0,0, 0,0,0, 0,0,0]
    ds_m_val=np.nan#ds_m_fun(delaoval,deloval,delmval,bval,p)
    if np.isnan(ds_m_val):
        ds_m_val=delmval

    points[:]=sorted([delaoval-deloval+delmval,delaoval-deloval+delmval+p['gamma2d'],delaoval-deloval+delmval-p['gamma2d'],
    delaoval-deloval+delmval+p['gamma2d']*70,delaoval-deloval+delmval-p['gamma2d']*70,
    delaoval-deloval+delmval+p['gamma2d']*200,delaoval-deloval+delmval-p['gamma2d']*200,
    delaoval-deloval+delmval+p['gamma2d']*700,delaoval-deloval+delmval-p['gamma2d']*700,
    delmval,delmval+p['gamma2d'],delmval+p['gamma2d']*100,delmval+p['gamma2d']*500,
    delmval-p['gamma2d'],delmval-p['gamma2d']*100,delmval-p['gamma2d']*500,
    -50*p['sd_delam']+p['mean_delam'],50*p['sd_delam']+p['mean_delam'],p['mean_delam'],
    p['sd_delam']+p['mean_delam'] ,-p['sd_delam']+p['mean_delam'],3*p['sd_delam']+p['mean_delam'] ,-3*p['sd_delam']+p['mean_delam'],
    10*p['sd_delam']+p['mean_delam'] ,-10*p['sd_delam']+p['mean_delam'],35*p['sd_delam']+p['mean_delam'] ,-35*p['sd_delam']+p['mean_delam'],
    delmval+p['gamma2d']*3,delmval+p['gamma2d']*10,delmval-p['gamma2d']*3,delmval-p['gamma2d']*10,0])
    for ii in range(len(points)-2):
      if (not points[ii]==points[ii+1]) and (np.isfinite(points[ii])):#and (not np.isinf(points[ii])):
        pointpairs[:]=[points[ii], points[ii+1]]
        S_temp=rho_m_broad_single(delaoval,deloval,delmval, p,pointpairs)
        #print(S_temp)
        S_out=[S_out[jj]+S_temp[jj] for jj in range(21)]
    S_temp=rho_m_broad_single_highbound(delaoval,deloval,delmval, p,points[len(points)-1])
    S_out=[S_out[jj]+S_temp[jj] for jj in range(21)]
    S_temp=rho_m_broad_single_lowbound(delaoval,deloval,delmval, p,points[0])
    S_out=[S_out[jj]+S_temp[jj] for jj in range(21)]
    return S_out

def rho_o_broad_single(double deloval,double delmval,ds_m_fun, p,delaolims):
    #xvals, wvals = _cached_roots_legendre(n)
    #yvals = (delaolims[1]-delaolims[0])*(xvals+1)/2.0 + delaolims[0]
    cdef double S_out[21]
    S_out[:]=np.zeros(21)#[0,0,0, 0,0,0 ,0,0,0]
    cdef int ii, jj
    cdef double rho_temp[21]
    cdef double yval
    cdef double gauss_weight
    for ii in range(n_gauss):
        yval=(delaolims[1]-delaolims[0])*(xgauss[ii]+1)/2.0 + delaolims[0]
        gauss_weight=gauss_fun_1d(yval,p['mean_delao'],p['sd_delao'])*wgauss[ii]
        rho_temp=rho_m_broad_full(yval,deloval,delmval,ds_m_fun,p)
        S_out=[S_out[jj]+rho_temp[jj]*(delaolims[1]-delaolims[0])/2.0*gauss_weight for jj in range(21)]
    return S_out

def rho_o_broad_single_highbound(double deloval,double delmval,ds_m_fun, p, double delaolim):
        #xvals, wvals = _cached_roots_legendre(n)
        #yvals = (delaolims[1]-delaolims[0])*(xvals+1)/2.0 + delaolims[0]
    cdef double S_out[21]
    S_out[:]=np.zeros(21)#[0,0,0, 0,0,0 ,0,0,0]
    cdef int ii, jj
    cdef double rho_temp[21]
    cdef double gaussexp_weight
    for ii in range(n_lag):
      gaussexp_weight=gauss_fun_1d(xlag[ii]+delaolim,p['mean_delao'],p['sd_delao'])*wlag[ii]*exp(xlag[ii])
      rho_temp=rho_m_broad_full(xlag[ii]+delaolim,deloval,delmval,ds_m_fun,p)
      S_out=[S_out[jj]+rho_temp[jj]*gaussexp_weight for jj in range(21)]
    return S_out

def rho_o_broad_single_lowbound(double deloval,double delmval,ds_m_fun, p, double delaolim):
    #xvals, wvals = _cached_roots_legendre(n)
    #yvals = (delaolims[1]-delaolims[0])*(xvals+1)/2.0 + delaolims[0]
    cdef double S_out[21]
    S_out[:]=np.zeros(21)#[0,0,0, 0,0,0 ,0,0,0]
    cdef int ii, jj
    cdef double rho_temp[21]
    cdef double gaussexp_weight
    for ii in range(n_lag):
        gaussexp_weight=gauss_fun_1d(-xlag[ii]+delaolim,p['mean_delao'],p['sd_delao'])*wlag[ii]*exp(xlag[ii])
        rho_temp=rho_m_broad_full(-xlag[ii]+delaolim,deloval,delmval,ds_m_fun,p)
        S_out=[S_out[jj]+rho_temp[jj]*gaussexp_weight for jj in range(21)]
    return S_out
def rho_broad_full(double deloval,double delmval,ds_m_fun, p):
    cdef double points[27]#[8]
    cdef double S_out_full[21]
    cdef double S_temp[21]
    cdef int ii,jj
    cdef double pointpairs[2]
    S_out_full=np.zeros(21)#[0,0,0, 0,0,0, 0,0,0]
    points[:]=sorted([-50*p['sd_delao']+p['mean_delao'],50*p['sd_delao']+p['mean_delao'],-500*p['sd_delao']+p['mean_delao'],500*p['sd_delao']+p['mean_delao'],
    -10*p['sd_delao']+p['mean_delao'],10*p['sd_delao']+p['mean_delao'],-35*p['sd_delao']+p['mean_delao'],35*p['sd_delao']+p['mean_delao'],
    -200*p['sd_delao']+p['mean_delao'],200*p['sd_delao']+p['mean_delao'],
    p['mean_delao'],-3*p['sd_delao']+p['mean_delao'],3*p['sd_delao']+p['mean_delao'],
    -p['sd_delao']+p['mean_delao'],p['sd_delao']+p['mean_delao'],
    deloval,deloval+p['gamma3d'],deloval+p['gamma3d']*5,deloval+p['gamma3d']*50,
    deloval-p['gamma3d'],deloval-p['gamma3d']*5,deloval-p['gamma3d']*50,
    deloval-p['gamma3d']*250,deloval+p['gamma3d']*250,deloval-p['gamma3d']*750,deloval+p['gamma3d']*750,0])
    #points[:]=sorted([p['mean_delao'],-3*p['sd_delao']+p['mean_delao'],3*p['sd_delao']+p['mean_delao'],-p['sd_delao']+p['mean_delao'],p['sd_delao']+p['mean_delao'],deloval,deloval+2*p['gamma3d'],deloval-2*p['gamma3d']])
    #print(type(points))
    #for ii in prange(9,nogil=True):

    for ii in range(len(points)-2):
        if not points[ii]==points[ii+1]:
          pointpairs[:]=[points[ii], points[ii+1]]
          S_temp=rho_o_broad_single(deloval,delmval,ds_m_fun, p,pointpairs)
          S_out_full=[S_out_full[jj]+S_temp[jj] for jj in range(21)]
    S_temp=rho_o_broad_single_highbound(deloval,delmval,ds_m_fun, p,points[len(points)-1])
    S_out_full=[S_out_full[jj]+S_temp[jj] for jj in range(21)]
    S_temp=rho_o_broad_single_lowbound(deloval,delmval,ds_m_fun, p,points[0])
    S_out_full=[S_out_full[jj]+S_temp[jj] for jj in range(21)]

    #S_out_fullpy=np.array(S_out_full)
    S_out_fullpy=np.zeros(45)
    for ii,newind in enumerate([ 0,  1,  2,  7,  8, 12, 13, 14, 15, 21, 22, 23, 24, 30, 31, 32, 33,39, 40, 41, 42]):
        S_out_fullpy[newind]=S_out_full[ii]
    #print(S_out_full)
    #print(S_out_fullpy)
    #return S_out_full
    # print('hon')
    # CtoRinv=np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
    #                   [0, 0, 0, 1, -1j, 0, 0, 0, 0],
    #                   [0, 0, 0, 0, 0, 1, -1j, 0, 0],
    #                   [0, 0, 0, 1, 1j, 0, 0, 0, 0],
    #                   [0, 1, 0, 0, 0, 0, 0, 0, 0],
    #                   [0, 0, 0, 0, 0, 0, 0, 1, -1j],
    #                   [0, 0, 0, 0, 0, 1, 1j, 0, 0],
    #                   [0, 0, 0, 0, 0, 0, 0, 1, 1j],
    #                   [0, 0, 1, 0, 0, 0, 0, 0, 0]])*-1 #-1 is because of the -1*L0^(-1)*Lx*rho0
    #
    # rho0=np.reshape(np.matmul(CtoRinv,(S_out_fullpy[0:9])),(3,3))
    # rhoar=np.reshape(np.matmul(CtoRinv,(S_out_fullpy[9:18])),(3,3))
    # rhoai=np.reshape(np.matmul(CtoRinv,(S_out_fullpy[18:27])),(3,3))
    # rhobr=np.reshape(np.matmul(CtoRinv,(S_out_fullpy[27:36])),(3,3))
    # rhobi=np.reshape(np.matmul(CtoRinv,(S_out_fullpy[36:45])),(3,3))
    # # rhoa=rhoar + 1j*rhoai
    # # rhoac=rhoar - 1j*rhoai
    # # rhob=rhobr + 1j*rhobi
    # # rhobc=rhobr - 1j*rhobi
    # rhoa=(rhoar - 1j*rhoai)/2
    # rhoac=(rhoar + 1j*rhoai)/2
    # rhob=(rhobr - 1j*rhobi)/2
    # rhobc=(rhobr + 1j*rhobi)/2
    CtoR = np.matrix([[2,0,0,0,0,0,0,0,0],
           [0,0,0,0,2,0,0,0,0],
           [0,0,0,0,0,0,0,0,2],
           [0,1,0,1,0,0,0,0,0],
           [0,1j,0,-1j,0,0,0,0,0],
           [0,0,1,0,0,0,1,0,0],
           [0,0,1j,0,0,0,-1j,0,0],
           [0,0,0,0,0,1,0,1,0],
           [0,0,0,0,0,1j,0,-1j,0]
          ])#*-1
    CtoR=CtoR/2
    CtoRinv=np.linalg.inv(CtoR)*-1#j
    rho0=np.reshape(CtoRinv*np.matrix(S_out_fullpy[0:9]).T,(3,3),order='C')
    rhoar=np.reshape(CtoRinv*np.matrix(S_out_fullpy[9:18]).T,(3,3),order='C')
    rhoai=np.reshape(CtoRinv*np.matrix(S_out_fullpy[18:27]).T,(3,3),order='C')
    rhobr=np.reshape(CtoRinv*np.matrix(S_out_fullpy[27:36]).T,(3,3),order='C')
    rhobi=np.reshape(CtoRinv*np.matrix(S_out_fullpy[36:45]).T,(3,3),order='C')
    # rho0=np.reshape(CtoRinv*np.matrix(S_out_fullpy[0:9]).T,(3,3),order='F')
    # rhoar=np.reshape(CtoRinv*np.matrix(S_out_fullpy[9:18]).T,(3,3),order='F')
    # rhoai=np.reshape(CtoRinv*np.matrix(S_out_fullpy[18:27]).T,(3,3),order='F')
    # rhobr=np.reshape(CtoRinv*np.matrix(S_out_fullpy[27:36]).T,(3,3),order='F')
    # rhobi=np.reshape(CtoRinv*np.matrix(S_out_fullpy[36:45]).T,(3,3),order='F')
    # rhoa=rhoar + 1j*rhoai
    # rhoac=rhoar - 1j*rhoai
    # rhob=rhobr + 1j*rhobi
    # rhobc=rhobr - 1j*rhobi
    rhoa=(rhoar - 1j*rhoai)/2
    rhoac=(rhoar + 1j*rhoai)/2
    rhob=(rhobr - 1j*rhobi)/2
    rhobc=(rhobr + 1j*rhobi)/2

    return rho0,rhoa,rhoac,rhob,rhobc


    #return S_out_full
    #return rho0,rhoar,rhoai,rhobr,rhobi
    #return [[S_out_full[0],S_out_full[3]+1j*S_out_full[4],S_out_full[5]+1j*S_out_full[6]],[S_out_full[3]-1j*S_out_full[4],S_out_full[1],S_out_full[7]+1j*S_out_full[8]],[S_out_full[5]-1j*S_out_full[6],S_out_full[7]-1j*S_out_full[8],S_out_full[2]]]

def rho_broad_full_real_imag(double deloval,double delmval,ds_m_fun, p):
    cdef double points[27]#[8]
    cdef double S_out_full[21]
    cdef double S_temp[21]
    cdef int ii,jj
    cdef double pointpairs[2]
    S_out_full=np.zeros(21)#[0,0,0, 0,0,0, 0,0,0]
    points[:]=sorted([-50*p['sd_delao']+p['mean_delao'],50*p['sd_delao']+p['mean_delao'],-500*p['sd_delao']+p['mean_delao'],500*p['sd_delao']+p['mean_delao'],
    -10*p['sd_delao']+p['mean_delao'],10*p['sd_delao']+p['mean_delao'],-35*p['sd_delao']+p['mean_delao'],35*p['sd_delao']+p['mean_delao'],
    -200*p['sd_delao']+p['mean_delao'],200*p['sd_delao']+p['mean_delao'],
    p['mean_delao'],-3*p['sd_delao']+p['mean_delao'],3*p['sd_delao']+p['mean_delao'],
    -p['sd_delao']+p['mean_delao'],p['sd_delao']+p['mean_delao'],
    deloval,deloval+p['gamma3d'],deloval+p['gamma3d']*5,deloval+p['gamma3d']*50,
    deloval-p['gamma3d'],deloval-p['gamma3d']*5,deloval-p['gamma3d']*50,
    deloval-p['gamma3d']*250,deloval+p['gamma3d']*250,deloval-p['gamma3d']*750,deloval+p['gamma3d']*750,0])
    #points[:]=sorted([p['mean_delao'],-3*p['sd_delao']+p['mean_delao'],3*p['sd_delao']+p['mean_delao'],-p['sd_delao']+p['mean_delao'],p['sd_delao']+p['mean_delao'],deloval,deloval+2*p['gamma3d'],deloval-2*p['gamma3d']])
    #print(type(points))
    #for ii in prange(9,nogil=True):

    for ii in range(len(points)-2):
        if not points[ii]==points[ii+1]:
          pointpairs[:]=[points[ii], points[ii+1]]
          S_temp=rho_o_broad_single(deloval,delmval,ds_m_fun, p,pointpairs)
          S_out_full=[S_out_full[jj]+S_temp[jj] for jj in range(21)]
    S_temp=rho_o_broad_single_highbound(deloval,delmval,ds_m_fun, p,points[len(points)-1])
    S_out_full=[S_out_full[jj]+S_temp[jj] for jj in range(21)]
    S_temp=rho_o_broad_single_lowbound(deloval,delmval,ds_m_fun, p,points[0])
    S_out_full=[S_out_full[jj]+S_temp[jj] for jj in range(21)]

    #S_out_fullpy=np.array(S_out_full)
    S_out_fullpy=np.zeros(45)
    for ii,newind in enumerate([ 0,  1,  2,  7,  8, 12, 13, 14, 15, 21, 22, 23, 24, 30, 31, 32, 33,39, 40, 41, 42]):
        S_out_fullpy[newind]=S_out_full[ii]
    #print(S_out_full)
    #print(S_out_fullpy)
    #return S_out_full
    # print('hon')
    # CtoRinv=np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
    #                   [0, 0, 0, 1, -1j, 0, 0, 0, 0],
    #                   [0, 0, 0, 0, 0, 1, -1j, 0, 0],
    #                   [0, 0, 0, 1, 1j, 0, 0, 0, 0],
    #                   [0, 1, 0, 0, 0, 0, 0, 0, 0],
    #                   [0, 0, 0, 0, 0, 0, 0, 1, -1j],
    #                   [0, 0, 0, 0, 0, 1, 1j, 0, 0],
    #                   [0, 0, 0, 0, 0, 0, 0, 1, 1j],
    #                   [0, 0, 1, 0, 0, 0, 0, 0, 0]])*-1 #-1 is because of the -1*L0^(-1)*Lx*rho0
    # rho0=np.reshape(np.matmul(CtoRinv,(S_out_fullpy[0:9])),(3,3),order='F')
    # rhoar=np.reshape(np.matmul(CtoRinv,(S_out_fullpy[9:18])),(3,3),order='F')
    # rhoai=np.reshape(np.matmul(CtoRinv,(S_out_fullpy[18:27])),(3,3),order='F')
    # rhobr=np.reshape(np.matmul(CtoRinv,(S_out_fullpy[27:36])),(3,3),order='F')
    # rhobi=np.reshape(np.matmul(CtoRinv,(S_out_fullpy[36:45])),(3,3),order='F')
    CtoR = np.matrix([[2,0,0,0,0,0,0,0,0],
           [0,0,0,0,2,0,0,0,0],
           [0,0,0,0,0,0,0,0,2],
           [0,1,0,1,0,0,0,0,0],
           [0,1j,0,-1j,0,0,0,0,0],
           [0,0,1,0,0,0,1,0,0],
           [0,0,1j,0,0,0,-1j,0,0],
           [0,0,0,0,0,1,0,1,0],
           [0,0,0,0,0,1j,0,-1j,0]
          ])
    CtoR=CtoR/2
    CtoRinv=np.linalg.inv(CtoR)*-1
    rho0=np.reshape(CtoRinv*np.matrix(S_out_fullpy[0:9]).T,(3,3),order='F')
    rhoar=np.reshape(CtoRinv*np.matrix(S_out_fullpy[9:18]).T,(3,3),order='F')
    rhoai=np.reshape(CtoRinv*np.matrix(S_out_fullpy[18:27]).T,(3,3),order='F')
    rhobr=np.reshape(CtoRinv*np.matrix(S_out_fullpy[27:36]).T,(3,3),order='F')
    rhobi=np.reshape(CtoRinv*np.matrix(S_out_fullpy[36:45]).T,(3,3),order='F')
    # rhoa=rhoar + 1j*rhoai
    # rhoac=rhoar - 1j*rhoai
    # rhob=rhobr + 1j*rhobi
    # rhobc=rhobr - 1j*rhobi
    rhoa=(rhoar - 1j*rhoai)/2
    rhoac=(rhoar + 1j*rhoai)/2
    rhob=(rhobr - 1j*rhobi)/2
    rhobc=(rhobr + 1j*rhobi)/2

    return rho0,rhoar,rhoai,rhobr,rhobi
