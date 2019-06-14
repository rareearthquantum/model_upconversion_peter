
cimport cython
import numpy as np
from cython.parallel cimport prange
cimport scipy.linalg.cython_lapack as lapack
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
    Lreal[3] =b_i*g_mu
    Lreal[4] =b_r*g_mu
    Lreal[5] =a_i*g_o
    Lreal[6] =a_r*g_o
    Lreal[7] =0
    Lreal[8] =0
    Lreal[9] =1
    Lreal[10] =-gamma_mu*(n_b + 1)
    Lreal[11] =0
    Lreal[12] =-b_i*g_mu
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
    Lreal[23] =-a_i*g_o
    Lreal[24] =-a_r*g_o
    Lreal[25] =0
    Lreal[26] =-Omega
    Lreal[27] =0
    Lreal[28] =2*b_i*g_mu
    Lreal[29] =0
    Lreal[30] =-gamma_2d/2 - gamma_mu*n_b - gamma_mu/2
    Lreal[31] =delta_a_mu - delta_mu
    Lreal[32] =0
    Lreal[33] =Omega
    Lreal[34] =a_i*g_o
    Lreal[35] =a_r*g_o
    Lreal[36] =0
    Lreal[37] =2*b_r*g_mu
    Lreal[38] =0
    Lreal[39] =-delta_a_mu + delta_mu
    Lreal[40] =-gamma_2d/2 - gamma_mu*n_b - gamma_mu/2
    Lreal[41] =-Omega
    Lreal[42] =0
    Lreal[43] =a_r*g_o
    Lreal[44] =-a_i*g_o
    Lreal[45] =0
    Lreal[46] =0
    Lreal[47] =2*a_i*g_o
    Lreal[48] =0
    Lreal[49] =Omega
    Lreal[50] =-gamma_13/2 - gamma_23/2 - gamma_3d/2 - gamma_mu*n_b/2
    Lreal[51] =delta_a_o - delta_o
    Lreal[52] =b_i*g_mu
    Lreal[53] =-b_r*g_mu
    Lreal[54] =0
    Lreal[55] =0
    Lreal[56] =2*a_r*g_o
    Lreal[57] =-Omega
    Lreal[58] =0
    Lreal[59] =-delta_a_o + delta_o
    Lreal[60] =-gamma_13/2 - gamma_23/2 - gamma_3d/2 - gamma_mu*n_b/2
    Lreal[61] =b_r*g_mu
    Lreal[62] =b_i*g_mu
    Lreal[63] =0
    Lreal[64] =0
    Lreal[65] =0
    Lreal[66] =-a_i*g_o
    Lreal[67] =-a_r*g_o
    Lreal[68] =-b_i*g_mu
    Lreal[69] =-b_r*g_mu
    Lreal[70] =-gamma_13/2 - gamma_23/2 - gamma_2d/2 - gamma_3d/2 - gamma_mu*(n_b + 1)/2
    Lreal[71] =-delta_a_mu + delta_a_o + delta_mu - delta_o
    Lreal[72] =0
    Lreal[73] =-2*Omega
    Lreal[74] =2*Omega
    Lreal[75] =-a_r*g_o
    Lreal[76] =a_i*g_o
    Lreal[77] =b_r*g_mu
    Lreal[78] =-b_i*g_mu
    Lreal[79] =delta_a_mu - delta_a_o - delta_mu + delta_o
    Lreal[80] =-gamma_13/2 - gamma_23/2 - gamma_2d/2 - gamma_3d/2 - gamma_mu*(n_b + 1)/2

    #rho = np.linalg.solve(L_mat.astype('complex128'),np.matrix([[1,0,0,0,0,0,0,0,0]]).T)
    lapack.dgesv(&n,&nrhs,Lreal,&lda,workspacec,V,&ldb, &info)
    #lapack.dgesv(&n,&nrhs,L,&lda,workspace,V,&ldb,&info)
    return V
    #return [V[3]+1j*V[4],V[5]+1j*V[6]]
    #return np.array([[V[0],V[3]+1j*V[4],V[5]+1j*V[6]],[V[3]-1j*V[4],V[1],V[7]+1j*V[8]],[V[5]-1j*V[6],V[7]-1j*V[8],V[2]]])

def gauss_fun_1d(double x,double m, double sd):
    return 1.0/(sqrt(2*3.14159265358979323846)*sd)*exp(-(x-m)**2/(2*sd**2)) #2.5066282746310002 is sqrt(2*pi)


DEF n_gauss=15#35
DEF n_lag = 15#35#15
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
    #cdef double points[17]#[6]
    cdef double points[23]#[6]

    cdef double S_out[9]
    cdef double S_temp[9]
    cdef double pointpairs[2]
    cdef double ds_m_val
    cdef int ii,jj
    cdef double omegamu = p['gm']*abs(bval)
    cdef double omegamu2 = omegamu**2
    cdef double deltaoval = delaoval-deloval
    if abs(deltaoval) <omegamu:
      deltaoval=copysign(omegamu,deltaoval)
    S_out=[0,0,0, 0,0,0, 0,0,0]
    ds_m_val=ds_m_fun(delaoval,deloval,delmval,bval,p)
    if np.isnan(ds_m_val):
        ds_m_val=delmval
    #points[:]=sorted([-50*p['sd_delam']+p['mean_delam'],50*p['sd_delam']+p['mean_delam'],p['mean_delam'],p['sd_delam']+p['mean_delam'] ,-p['sd_delam']+p['mean_delam'],delmval,delmval+2*p['gamma2d']
    #,delmval-2*p['gamma2d'],ds_m_val, ds_m_val+2*p['gammamu'],ds_m_val-2*p['gammamu'], ds_m_val+2*p['gamma2d'],ds_m_val-2*p['gamma2d'],delaoval-delmval]) #,p['sd_delam']+p['mean_delam']]
    #points[:]=sorted([p['mean_delam'],p['sd_delam']+p['mean_delam'] ,-p['sd_delam']+p['mean_delam'],delmval,delmval+2*p['gamma2d'],delmval-2*p['gamma2d']])#,p['sd_delam']+p['mean_delam']]
    # points[:]=sorted([-50*p['sd_delam']+p['mean_delam'],50*p['sd_delam']+p['mean_delam'],p['mean_delam'],p['sd_delam']+p['mean_delam'] ,-p['sd_delam']+p['mean_delam'],delmval,delmval+2*p['gamma2d']
    # ,delmval-2*p['gamma2d'],ds_m_val, ds_m_val+2*p['gammamu'],ds_m_val-2*p['gammamu'], ds_m_val+2*p['gamma2d'],ds_m_val-2*p['gamma2d'], #,p['sd_delam']+p['mean_delam']]
    # -omegamu2/(delaoval-deloval)*2+delmval,delaoval-deloval+delmval,(delaoval-deloval)/2+np.sqrt((delaoval-deloval)**2/4-omegamu2)+delmval,(delaoval-deloval)/2-np.sqrt((delaoval-deloval)**2/4-omegamu2)+delmval])
    points[:]=sorted([-50*p['sd_delam']+p['mean_delam'],50*p['sd_delam']+p['mean_delam'],p['mean_delam'],p['sd_delam']+p['mean_delam'] ,-p['sd_delam']+p['mean_delam'],delmval,delmval+2*p['gamma2d']
    ,delmval-2*p['gamma2d'],ds_m_val, ds_m_val+2*p['gammamu'],ds_m_val-2*p['gammamu'], ds_m_val+2*p['gamma2d'],ds_m_val-2*p['gamma2d'],delaoval-deloval+delmval, #,p['sd_delam']+p['mean_delam']]
    -omegamu2/(deltaoval )*2+delmval,(delaoval-deloval)/2+np.sqrt((delaoval-deloval)**2/4-omegamu2)+delmval,(delaoval-deloval)/2-np.sqrt((delaoval-deloval)**2/4-omegamu2)+delmval,
    -omegamu2/(deltaoval )*2+delmval-omegamu,(delaoval-deloval)/2+np.sqrt((delaoval-deloval)**2/4-omegamu2)+delmval-omegamu,(delaoval-deloval)/2-np.sqrt((delaoval-deloval)**2/4-omegamu2)+delmval-omegamu,
    -omegamu2/(deltaoval )*2+delmval+omegamu,(delaoval-deloval)/2+np.sqrt((delaoval-deloval)**2/4-omegamu2)+delmval+omegamu,(delaoval-deloval)/2-np.sqrt((delaoval-deloval)**2/4-omegamu2)+delmval+omegamu])

    for ii in range(len(points)-1):
      if (not points[ii]==points[ii+1]) and (np.isfinite(points[ii])):#and (not np.isinf(points[ii])):
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
    for ii in range(9):
        if not points[ii]==points[ii+1]:
          pointpairs[:]=[points[ii], points[ii+1]]
          S_temp=rho_o_broad_single(aval,bval,deloval,delmval,ds_m_fun, p,pointpairs)
          S_out_full=[S_out_full[jj]+S_temp[jj] for jj in range(9)]
    S_temp=rho_o_broad_single_highbound(aval,bval,deloval,delmval,ds_m_fun, p,points[9])
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
