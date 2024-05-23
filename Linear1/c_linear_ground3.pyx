
cimport cython
import numpy as np
from cython.parallel cimport prange
cimport scipy.linalg.cython_lapack as lapack
from scipy.linalg.cython_blas cimport dgemm

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

def steady_rhos(double delta_a_o, double delta_a_mu,double delta_o,double delta_mu,p):

    cdef double L0[81]
    cdef double rhovec[21]
    #cdef double Lar[81]
    #cdef double Lai[81]
    #cdef double Lbr[81]
    #cdef double Lai[81]
    cdef int ii = 0
    cdef int jj
    cdef int rho0inds[5] #= [0,1,2,7,8]
    cdef int rhoxinds[4] #= [3,4,5,6]
    rho0inds[:] = [0,1,2,7,8]
    rhoxinds[:] = [3,4,5,6]
    cdef double V[9]
    cdef double rho0[9]
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
    Lar=([[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 2*g_o, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, -g_o], [0, 0, 0, 0, 0, 0, 0, -g_o, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [g_o, 0, -g_o, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, g_o, 0, 0, 0, 0], [0, 0, 0, g_o, 0, 0, 0, 0, 0]])
    Lai=([[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 2*g_o, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, -g_o, 0], [0, 0, 0, 0, 0, 0, 0, 0, g_o], [g_o, 0, -g_o, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, g_o, 0, 0, 0, 0, 0], [0, 0, 0, 0, -g_o, 0, 0, 0, 0]])
    Lbr=([[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 2*g_mu, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [g_mu, -g_mu, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, g_mu], [0, 0, 0, 0, 0, 0, 0, -g_mu, 0], [0, 0, 0, 0, 0, 0, g_mu, 0, 0], [0, 0, 0, 0, 0, -g_mu, 0, 0, 0]])
    Lbi=([[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 2*g_mu, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [g_mu, -g_mu, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, -g_mu, 0], [0, 0, 0, 0, 0, 0, 0, 0, -g_mu], [0, 0, 0, 0, 0, g_mu, 0, 0, 0], [0, 0, 0, 0, 0, 0, g_mu, 0, 0]])
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
    L0[70] = -gamma_13/2 - gamma_23/2 - gamma_2d/2 - gamma_3d/2 - gamma_mu*(n_b + 1)/2
    L0[71] = -delta_a_mu + delta_a_o + delta_mu - delta_o
    L0[72] = 0
    L0[73] = -2*Omega
    L0[74] = 2*Omega
    L0[75] = 0
    L0[76] = 0
    L0[77] = 0
    L0[78] = 0
    L0[79] = delta_a_mu - delta_a_o - delta_mu + delta_o
    L0[80] = -gamma_13/2 - gamma_23/2 - gamma_2d/2 - gamma_3d/2 - gamma_mu*(n_b + 1)/2

    #rhovec=np.zeros(45)
    rho0[:]=[1,0,0, 0,0,0, 0,0,0]
    #for ii in range(9):
    #  rhovec[
    #rho = np.linalg.solve(L_mat.astype('complex128'),np.matrix([[1,0,0,0,0,0,0,0,0]]).T)
    lapack.dgesv(&n,&nrhs,L0,&lda,workspacec,rho0,&ldb, &info)
    #rhovec[0:9]=rho0#[:]
    for jj in range(5):
      rhovec[jj]=rho0[rho0inds[jj]]
    for Lx in [Lar, Lai, Lbr, Lbi]:
      V[:]=np.matmul(Lx,rho0)
      lapack.dgesv(&n,&nrhs,L0,&lda,workspacec,V,&ldb, &info)
      for jj in range(4):
        rhovec[5+jj+4*ii]=V[rhoxinds[jj]]
        #print(5+jj+4*ii)
      #rhovec[(9+ii*9):(18+ii*9)]=V#np.array(V[:])
      #print(V)
      #print(str(ii) + '  '+ str(np.where(np.array(V)==0)[0]))
      #print(str(ii) + '  '+ str(np.nonzero(np.array(V))))
      ii=ii+1
    #lapack.dgesv(&n,&nrhs,L,&lda,workspace,V,&ldb,&info)
    #print(str(np.nonzero(np.array(rho0))))
    #print(str(np.where(np.array(rho0)==0)[0]))

    return rhovec
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
#    for ii in range(len(points)-1):
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
    cdef double points[14]#[6]

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
    points[:]=sorted([delaoval-deloval+delmval,delaoval-deloval+delmval+p['gamma2d'],delaoval-deloval+delmval-p['gamma2d'],delaoval-deloval+delmval+p['gamma2d']*70,delaoval-deloval+delmval-p['gamma2d']*70,delaoval-deloval+delmval+p['gamma2d']*200,delaoval-deloval+delmval-p['gamma2d']*200, delaoval-deloval+delmval+p['gamma2d']*700,delaoval-deloval+delmval-p['gamma2d']*700, -50*p['sd_delam']+p['mean_delam'],50*p['sd_delam']+p['mean_delam'],p['mean_delam'],p['sd_delam']+p['mean_delam'] ,-p['sd_delam']+p['mean_delam']])
    for ii in range(len(points)-1):
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
    cdef double points[10]#[8]
    cdef double S_out_full[21]
    cdef double S_temp[21]
    cdef int ii,jj
    cdef double pointpairs[2]
    S_out_full=np.zeros(21)#[0,0,0, 0,0,0, 0,0,0]
    points[:]=sorted([-50*p['sd_delao']+p['mean_delao'],50*p['sd_delao']+p['mean_delao'],p['mean_delao'],-3*p['sd_delao']+p['mean_delao'],3*p['sd_delao']+p['mean_delao'],-p['sd_delao']+p['mean_delao']
    ,p['sd_delao']+p['mean_delao'],deloval,deloval+2*p['gamma3d'],deloval-2*p['gamma3d']])
    #points[:]=sorted([p['mean_delao'],-3*p['sd_delao']+p['mean_delao'],3*p['sd_delao']+p['mean_delao'],-p['sd_delao']+p['mean_delao'],p['sd_delao']+p['mean_delao'],deloval,deloval+2*p['gamma3d'],deloval-2*p['gamma3d']])
    #print(type(points))
    #for ii in prange(9,nogil=True):

    for ii in range(9):
        if not points[ii]==points[ii+1]:
          pointpairs[:]=[points[ii], points[ii+1]]
          S_temp=rho_o_broad_single(deloval,delmval,ds_m_fun, p,pointpairs)
          S_out_full=[S_out_full[jj]+S_temp[jj] for jj in range(21)]
    S_temp=rho_o_broad_single_highbound(deloval,delmval,ds_m_fun, p,points[9])
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
    CtoRinv=np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, -1j, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, -1j, 0, 0],
                      [0, 0, 0, 1, 1j, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, -1j],
                      [0, 0, 0, 0, 0, 1, 1j, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 1j],
                      [0, 0, 1, 0, 0, 0, 0, 0, 0]])

    rho0=np.reshape(np.matmul(CtoRinv,(S_out_fullpy[0:9])),(3,3))
    rhoar=np.reshape(np.matmul(CtoRinv,(S_out_fullpy[9:18])),(3,3))
    rhoai=np.reshape(np.matmul(CtoRinv,(S_out_fullpy[18:27])),(3,3))
    rhobr=np.reshape(np.matmul(CtoRinv,(S_out_fullpy[27:36])),(3,3))
    rhobi=np.reshape(np.matmul(CtoRinv,(S_out_fullpy[36:45])),(3,3))
    rhoa=rhoar + 1j*rhoai
    rhoac=rhoar - 1j*rhoai
    rhob=rhobr + 1j*rhobi
    rhobc=rhobr - 1j*rhobi
    return rho0,rhoa,rhoac,rhob,rhobc
    #return S_out_full
    #return rho0,rhoar,rhoai,rhobr,rhobi
    #return [[S_out_full[0],S_out_full[3]+1j*S_out_full[4],S_out_full[5]+1j*S_out_full[6]],[S_out_full[3]-1j*S_out_full[4],S_out_full[1],S_out_full[7]+1j*S_out_full[8]],[S_out_full[5]-1j*S_out_full[6],S_out_full[7]-1j*S_out_full[8],S_out_full[2]]]
