cdef extern from "csteady.c":
      double complex csteady_rho31(double complex Omega_m,
			    double complex Omega_o,
			    double complex Omega_sb,
			    double delta_2,
			    double delta_3,
                            double gamma_13,
			    double gamma_23,
			    double gamma_2d,
			    double gamma_3d,
                            double n_b,
			    double gamma_m)
     

def steady_rho31(double complex Rabio,
    		double complex Rabim,
		double complex Rabisb,
		double delta3,
		double delta2,
		paramdict):

    
    return csteady_rho31( Rabim,
			  Rabio,
			  Rabisb,
			     delta2,
			     delta3,
                             paramdict['gamma_13'],
			     paramdict['gamma_23'],
			     paramdict['gamma_2d'],
			     paramdict['gamma_3d'],
                             paramdict['n_b'],
			     paramdict['gamma_m'])
    