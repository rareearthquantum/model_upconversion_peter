#include <math.h>
#include <complex.h>
#include <lapacke.h>



static double complex csteady_rho31(double complex Omega_m,
			    double complex Omega_o,
			    double complex Omega_sb,
			    double delta_2,
			    double delta_3,
                            double gamma_13,
			    double gamma_23,
			    double gamma_2d,
			    double gamma_3d,
                            double n_b,
			    double gamma_m){

    lapack_int info;
    



    
    complex double V[9] = {1,0,0, 0,0,0, 0,0,0};
    complex double L[81] = {1, -I*Omega_m, -I*Omega_sb, I*conj(Omega_m), gamma_m*n_b, 0, I*conj(Omega_sb), 0, 0, 0, -I*delta_2 - 1.0/2.0*gamma_2d - 1.0/2.0*gamma_m*n_b - 1.0/2.0*gamma_m*(n_b + 1), -I*Omega_o, 0, I*conj(Omega_m), 0, 0, I*conj(Omega_sb), 0, 0, -I*conj(Omega_o), -I*delta_3 - 1.0/2.0*gamma_13 - 1.0/2.0*gamma_23 - 1.0/2.0*gamma_3d - 1.0/2.0*gamma_m*n_b, 0, 0, I*conj(Omega_m), 0, 0, I*conj(Omega_sb), 0, 0, 0, I*delta_2 - 1.0/2.0*gamma_2d - 1.0/2.0*gamma_m*n_b - 1.0/2.0*gamma_m*(n_b + 1), -I*Omega_m, -I*Omega_sb, I*conj(Omega_o), 0, 0, 1, I*Omega_m, 0, -I*conj(Omega_m), -gamma_m*(n_b + 1), -I*Omega_o, 0, I*conj(Omega_o), 0, 0, 0, I*Omega_m, -I*conj(Omega_sb), -I*conj(Omega_o), I*delta_2 - I*delta_3 - 1.0/2.0*gamma_13 - 1.0/2.0*gamma_23 - 1.0/2.0*gamma_2d - 1.0/2.0*gamma_3d - 1.0/2.0*gamma_m*(n_b + 1), 0, 0, I*conj(Omega_o), 0, 0, 0, I*Omega_o, 0, 0, I*delta_3 - 1.0/2.0*gamma_13 - 1.0/2.0*gamma_23 - 1.0/2.0*gamma_3d - 1.0/2.0*gamma_m*n_b, -I*Omega_m, -I*Omega_sb, 0, I*Omega_sb, 0, 0, I*Omega_o, 0, -I*conj(Omega_m), -I*delta_2 + I*delta_3 - 1.0/2.0*gamma_13 - 1.0/2.0*gamma_23 - 1.0/2.0*gamma_2d - 1.0/2.0*gamma_3d - 1.0/2.0*gamma_m*(n_b + 1), -I*Omega_o, 1, 0, I*Omega_sb, 0, gamma_23, I*Omega_o, -I*conj(Omega_sb), -I*conj(Omega_o), -gamma_13 - gamma_23};

    lapack_int workspace[9];


    //  work out V\L
//    lapack_int LAPACKE_zgesv (int matrix_layout , lapack_int n , lapack_int nrhs , lapack_complex_double * a , lapack_int lda , lapack_int * ipiv , lapack_complex_double * b , lapack_int ldb );
    info = LAPACKE_zgesv(LAPACK_COL_MAJOR, 9, //n
			 1, //nrhs
			 L, //a
			 9, //lda
			 workspace, //ipiv
			 V, //b
			 9 //ldb
	);
    return V[6]; //rho31
    
}

main(){
    return 0;
}
