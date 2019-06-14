
cimport cython
import numpy as np
from cython.parallel cimport prange
cimport scipy.linalg.cython_lapack as lapack
#import numpy as np
from libc.math cimport exp, sqrt
import scipy.special#.roots_legendre


def steady_rho_single_c(double delta_a_o, double delta_a_mu, double complex aval,double complex bval,double delta_o,double delta_mu,p):

    cdef double Lreal[81]
    cdef double V[9]
    cdef int workspacec[9]
    cdef int info
    cdef int lda = 9
    cdef int ldb = 9
    cdef int nrhs = 1
    cdef int n = 9
    #here I am defining the detunings using values for the ground state system
    #THIS IS NOT HOW THE FULL CODE SHOULD WORK
    cdef double gamma_13 = p['gamma13']
    cdef double gamma_23 = p['gamma23']
    cdef double gamma_2d = p['gamma2d']
    cdef double gamma_3d = p['gamma3d']
    cdef double n_b = p['nbath']
    cdef double gamma_12 = p['gamma12']
    cdef double g_mu = p['gm']
    cdef double g_o = p['go']
    cdef double Omega = p['Omega']
    cdef double a_r=aval.real
    cdef double a_i=aval.imag
    cdef double b_r=bval.real
    cdef double b_i=bval.imag
    V[:]=[1,0,0, 0,0,0, 0,0,0]

    Lreal[0] =1
    Lreal[1] =0
    Lreal[2] =0
    Lreal[3] =0
    Lreal[4] =Omega
    Lreal[5] =a_i*g_o
    Lreal[6] =a_r*g_o
    Lreal[7] =0
    Lreal[8] =0
    Lreal[9] =1
    Lreal[10] =-gamma_12 - gamma_23*n_b
    Lreal[11] =gamma_23*n_b
    Lreal[12] =0
    Lreal[13] =-Omega
    Lreal[14] =0
    Lreal[15] =0
    Lreal[16] =b_i*g_mu
    Lreal[17] =b_r*g_mu
    Lreal[18] =1
    Lreal[19] =gamma_23*(n_b + 1)
    Lreal[20] =-gamma_13 - gamma_23*(n_b + 1)
    Lreal[21] =0
    Lreal[22] =0
    Lreal[23] =-a_i*g_o
    Lreal[24] =-a_r*g_o
    Lreal[25] =-b_i*g_mu
    Lreal[26] =-b_r*g_mu
    Lreal[27] =0
    Lreal[28] =0
    Lreal[29] =0
    Lreal[30] =-gamma_12/2 - gamma_23*n_b/2 - gamma_2d/2
    Lreal[31] =delta_a_mu - delta_mu
    Lreal[32] =b_i*g_mu
    Lreal[33] =b_r*g_mu
    Lreal[34] =a_i*g_o
    Lreal[35] =a_r*g_o
    Lreal[36] =0
    Lreal[37] =2*Omega
    Lreal[38] =0
    Lreal[39] =-delta_a_mu + delta_mu
    Lreal[40] =-gamma_12/2 - gamma_23*n_b/2 - gamma_2d/2
    Lreal[41] =-b_r*g_mu
    Lreal[42] =b_i*g_mu
    Lreal[43] =a_r*g_o
    Lreal[44] =-a_i*g_o
    Lreal[45] =0
    Lreal[46] =0
    Lreal[47] =2*a_i*g_o
    Lreal[48] =-b_i*g_mu
    Lreal[49] =b_r*g_mu
    Lreal[50] =-gamma_13/2 - gamma_23*(n_b + 1)/2 - gamma_3d/2
    Lreal[51] =delta_a_o - delta_o
    Lreal[52] =0
    Lreal[53] =-Omega
    Lreal[54] =0
    Lreal[55] =0
    Lreal[56] =2*a_r*g_o
    Lreal[57] =-b_r*g_mu
    Lreal[58] =-b_i*g_mu
    Lreal[59] =-delta_a_o + delta_o
    Lreal[60] =-gamma_13/2 - gamma_23*(n_b + 1)/2 - gamma_3d/2
    Lreal[61] =Omega
    Lreal[62] =0
    Lreal[63] =0
    Lreal[64] =-2*b_i*g_mu
    Lreal[65] =2*b_i*g_mu
    Lreal[66] =-a_i*g_o
    Lreal[67] =-a_r*g_o
    Lreal[68] =0
    Lreal[69] =-Omega
    Lreal[70] =-gamma_12/2 - gamma_13/2 - gamma_23*n_b - gamma_23/2 - gamma_2d/2 - gamma_3d/2
    Lreal[71] =-delta_a_mu + delta_a_o + delta_mu - delta_o
    Lreal[72] =0
    Lreal[73] =-2*b_r*g_mu
    Lreal[74] =2*b_r*g_mu
    Lreal[75] =-a_r*g_o
    Lreal[76] =a_i*g_o
    Lreal[77] =Omega
    Lreal[78] =0
    Lreal[79] =delta_a_mu - delta_a_o - delta_mu + delta_o
    Lreal[80] =-gamma_12/2 - gamma_13/2 - gamma_23*n_b - gamma_23/2 - gamma_2d/2 - gamma_3d/2

    #rho = np.linalg.solve(L_mat.astype('complex128'),np.matrix([[1,0,0,0,0,0,0,0,0]]).T)
    lapack.dgesv(&n,&nrhs,Lreal,&lda,workspacec,V,&ldb, &info)
    #lapack.dgesv(&n,&nrhs,L,&lda,workspace,V,&ldb,&info)
    return V
    #return [V[3]+1j*V[4],V[5]+1j*V[6]]
    #return np.array([[V[0],V[3]+1j*V[4],V[5]+1j*V[6]],[V[3]-1j*V[4],V[1],V[7]+1j*V[8]],[V[5]-1j*V[6],V[7]-1j*V[8],V[2]]])

def gauss_fun_1d(double x,double m, double sd):
    return 1.0/(sqrt(2*3.14159265358979323846)*sd)*exp(-(x-m)**2/(2*sd**2)) #2.5066282746310002 is sqrt(2*pi)


DEF n_gauss=15#30
DEF n_lag = 15#30#15
#global xgauss, wgauss
cdef double xgauss[n_gauss]
cdef double wgauss[n_gauss]
xgauss[:],wgauss[:]=scipy.special.roots_legendre(n_gauss)
cdef double xlag[n_lag]
cdef double wlag[n_lag]
xlag[:],wlag[:]=scipy.special.roots_laguerre(n_lag)


def rho_m_broad_single(double delaoval,double complex aval, double complex bval, double deloval,double delmval, p,delamlims):
    cdef double delamlimsc[2]
    delamlimsc[:]=delamlims
    #xvals, wvals = _cached_roots_legendre(n)
    cdef double yval#s[n_gauss]
    cdef double S_out[9]
    S_out[:]=[0,0,0, 0,0,0 ,0,0,0]
    cdef double rho_temp[9]
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
        rho_temp=(steady_rho_single_c(delaoval,yval,aval, bval,deloval,delmval,p))
        #S_out=[S_out[jj]+rho_temp[jj]*(delamlims[1]-delamlims[0])/2.0*gauss_weight for jj in range(9)]
        for jj in range(9):
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
#    for ii in range(len(points)-1):
#       S_out_full=S_out_full+rho_m_broad_single(delaoval,aval,bval,deloval,delmval, p,[points[ii], points[ii+1]],n=n)
#    return S_out_full

def rho_m_broad_single_highbound(double delaoval,double complex aval, double complex bval, double deloval,double delmval, p,double delamlim):
    #xvals, wvals = _cached_roots_legendre(n)
    #cdef double yval#s[n_gauss]
    cdef double S_out[9]
    S_out[:]=[0,0,0, 0,0,0 ,0,0,0]
    cdef double rho_temp[9]
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
        rho_temp=(steady_rho_single_c(delaoval,xlag[ii]+delamlim,aval, bval,deloval,delmval,p))
        #S_out=[S_out[jj]+rho_temp[jj]*(delamlims[1]-delamlims[0])/2.0*gauss_weight for jj in range(9)]
        for jj in range(9):
            S_out[jj]+=rho_temp[jj]*gaussexp_weight
    #S_out[:]=[0,0,0, 0,0,0 ,0,0,0]
    return S_out
def rho_m_broad_single_lowbound(double delaoval,double complex aval, double complex bval, double deloval,double delmval, p,double delamlim):
  #xvals, wvals = _cached_roots_legendre(n)
  #cdef double yval#s[n_gauss]
  cdef double S_out[9]
  S_out[:]=[0,0,0, 0,0,0 ,0,0,0]
  cdef double rho_temp[9]
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
      rho_temp=(steady_rho_single_c(delaoval,-xlag[ii]+delamlim,aval, bval,deloval,delmval,p))
      #S_out=[S_out[jj]+rho_temp[jj]*(delamlims[1]-delamlims[0])/2.0*gauss_weight for jj in range(9)]
      for jj in range(9):
          S_out[jj]+=rho_temp[jj]*gaussexp_weight
  #S_out[:]=[0,0,0, 0,0,0 ,0,0,0]
  return S_out
def rho_m_broad_full(double delaoval,double complex aval, double complex bval,double deloval,double delmval,ds_m_fun, p):
    cdef double points[14]#[6]
    cdef double S_out[9]
    cdef double S_temp[9]
    cdef double pointpairs[2]
    cdef double ds_m_val
    cdef int ii,jj
    S_out=[0,0,0, 0,0,0, 0,0,0]
    ds_m_val=ds_m_fun(delaoval,deloval,delmval,bval,p)
    if np.isnan(ds_m_val):
        ds_m_val=delmval
    points[:]=sorted([-50*p['sd_delam']+p['mean_delam'],50*p['sd_delam']+p['mean_delam'],p['mean_delam'],p['sd_delam']+p['mean_delam'] ,-p['sd_delam']+p['mean_delam'],
                          delmval,delmval+2*p['gamma2d'],delmval-2*p['gamma2d'],delaoval-delmval,delaoval-delmval-abs(p['go']*bval),delaoval-delmval+abs(p['go']*bval)
                          ,ds_m_val,ds_m_val+abs(p['go']*bval),ds_m_val-abs(p['go']*bval)])
    #points[:]=sorted([p['mean_delam'],p['sd_delam']+p['mean_delam'] ,-p['sd_delam']+p['mean_delam'],delmval,delmval+2*p['gamma2d'],delmval-2*p['gamma2d']])#,p['sd_delam']+p['mean_delam']]
    for ii in range(len(points)-1):
      if not points[ii]==points[ii+1]:
        pointpairs[:]=[points[ii], points[ii+1]]
        S_temp=rho_m_broad_single(delaoval,aval,bval,deloval,delmval, p,pointpairs)
        #print(S_temp)
        S_out=[S_out[jj]+S_temp[jj] for jj in range(9)]
    S_temp=rho_m_broad_single_highbound(delaoval,aval,bval,deloval,delmval, p,points[len(points)-1])
    S_out=[S_out[jj]+S_temp[jj] for jj in range(9)]
    S_temp=rho_m_broad_single_lowbound(delaoval,aval,bval,deloval,delmval, p,points[0])
    S_out=[S_out[jj]+S_temp[jj] for jj in range(9)]
    return S_out

def rho_o_broad_single(double complex aval,double complex bval,double deloval,double delmval,ds_m_fun, p,delaolims):
    #xvals, wvals = _cached_roots_legendre(n)
    #yvals = (delaolims[1]-delaolims[0])*(xvals+1)/2.0 + delaolims[0]
    cdef double S_out[9]
    S_out[:]=[0,0,0, 0,0,0 ,0,0,0]
    cdef int ii, jj
    cdef double rho_temp[9]
    cdef double yval
    cdef double gauss_weight
    for ii in range(n_gauss):
        yval=(delaolims[1]-delaolims[0])*(xgauss[ii]+1)/2.0 + delaolims[0]
        gauss_weight=gauss_fun_1d(yval,p['mean_delao'],p['sd_delao'])*wgauss[ii]
        rho_temp=rho_m_broad_full(yval,aval, bval,deloval,delmval,ds_m_fun,p)
        S_out=[S_out[jj]+rho_temp[jj]*(delaolims[1]-delaolims[0])/2.0*gauss_weight for jj in range(9)]
    return S_out

def rho_o_broad_single_highbound(double complex aval,double complex bval,double deloval,double delmval,ds_m_fun, p, double delaolim):
        #xvals, wvals = _cached_roots_legendre(n)
        #yvals = (delaolims[1]-delaolims[0])*(xvals+1)/2.0 + delaolims[0]
    cdef double S_out[9]
    S_out[:]=[0,0,0, 0,0,0 ,0,0,0]
    cdef int ii, jj
    cdef double rho_temp[9]
    cdef double gaussexp_weight
    for ii in range(n_lag):
      gaussexp_weight=gauss_fun_1d(xlag[ii]+delaolim,p['mean_delao'],p['sd_delao'])*wlag[ii]*exp(xlag[ii])
      rho_temp=rho_m_broad_full(xlag[ii]+delaolim,aval, bval,deloval,delmval,ds_m_fun,p)
      S_out=[S_out[jj]+rho_temp[jj]*gaussexp_weight for jj in range(9)]
    return S_out

def rho_o_broad_single_lowbound(double complex aval,double complex bval,double deloval,double delmval,ds_m_fun, p, double delaolim):
    #xvals, wvals = _cached_roots_legendre(n)
    #yvals = (delaolims[1]-delaolims[0])*(xvals+1)/2.0 + delaolims[0]
    cdef double S_out[9]
    S_out[:]=[0,0,0, 0,0,0 ,0,0,0]
    cdef int ii, jj
    cdef double rho_temp[9]
    cdef double gaussexp_weight
    for ii in range(n_lag):
        gaussexp_weight=gauss_fun_1d(-xlag[ii]+delaolim,p['mean_delao'],p['sd_delao'])*wlag[ii]*exp(xlag[ii])
        rho_temp=rho_m_broad_full(-xlag[ii]+delaolim,aval, bval,deloval,delmval,ds_m_fun,p)
        S_out=[S_out[jj]+rho_temp[jj]*gaussexp_weight for jj in range(9)]
    return S_out
def rho_broad_full(double complex aval,double complex bval,double deloval,double delmval,ds_m_fun, p):
    cdef double points[10]#[8]
    cdef double S_out_full[9]
    cdef double S_temp[9]
    cdef int ii,jj
    cdef double pointpairs[2]
    S_out_full=[0,0,0, 0,0,0, 0,0,0]
    points[:]=sorted([-50*p['sd_delao']+p['mean_delao'],50*p['sd_delao']+p['mean_delao'],p['mean_delao'],-3*p['sd_delao']+p['mean_delao'],3*p['sd_delao']+p['mean_delao'],-p['sd_delao']+p['mean_delao']
    ,p['sd_delao']+p['mean_delao'],deloval,deloval+2*p['gamma3d'],deloval-2*p['gamma3d']])
    #points[:]=sorted([p['mean_delao'],-3*p['sd_delao']+p['mean_delao'],3*p['sd_delao']+p['mean_delao'],-p['sd_delao']+p['mean_delao'],p['sd_delao']+p['mean_delao'],deloval,deloval+2*p['gamma3d'],deloval-2*p['gamma3d']])
    #print(type(points))
    #for ii in prange(9,nogil=True):
    for ii in range(len(points)-1):
        if not points[ii]==points[ii+1]:
          pointpairs[:]=[points[ii], points[ii+1]]
          S_temp=rho_o_broad_single(aval,bval,deloval,delmval,ds_m_fun, p,pointpairs)
          S_out_full=[S_out_full[jj]+S_temp[jj] for jj in range(9)]
    S_temp=rho_o_broad_single_highbound(aval,bval,deloval,delmval,ds_m_fun, p,points[len(points)-1])
    S_out_full=[S_out_full[jj]+S_temp[jj] for jj in range(9)]
    S_temp=rho_o_broad_single_lowbound(aval,bval,deloval,delmval,ds_m_fun, p,points[0])
    S_out_full=[S_out_full[jj]+S_temp[jj] for jj in range(9)]
    #return S_out_full
    return [[S_out_full[0],S_out_full[3]+1j*S_out_full[4],S_out_full[5]+1j*S_out_full[6]],[S_out_full[3]-1j*S_out_full[4],S_out_full[1],S_out_full[7]+1j*S_out_full[8]],[S_out_full[5]-1j*S_out_full[6],S_out_full[7]-1j*S_out_full[8],S_out_full[2]]]
#def rho_integrate_full(double complex aval, double complex bval, double deloval, delmval,p):#, int n=15):
    #cdef int n=15
 #   cdef double points[10]
  #  cdef double complex S_out_full[9]
   # cdef double xvals[n_gauss]
    #cdef double yvals[n_gauss]
    #cdef double wvals[n_gauss]
    #cdef int ii
    #xvals[:],wvals[:] = scipy.special.roots_legendre(n_gauss)#_cached_roots_legendre(n_gauss)
    #points[:]=[-50*p['sd_delao']+p['mean_delao'],50*p['sd_delao']+p['mean_delao'],p['mean_delao'],-3*p['sd_delao']+p['mean_delao'],3*p['sd_delao']+p['mean_delao'],-p['sd_delao']+p['mean_delao'],p['sd_delao']+p['mean_delao'],deloval,deloval+2*p['gamma3d'],deloval-2*p['gamma3d']]
    #points.sort()
    #for ii in prange(9,nogil=True):
    #    1+1
